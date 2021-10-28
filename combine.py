import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pylab as pl
import os
from scipy import signal,interpolate
from scipy.signal import find_peaks
from scipy.fftpack import fft,ifft
from scipy.signal import butter, lfilter
import random
from collections import deque
import serial
from mmWave import vitalsign
import time
import vitalsign_v2
import datetime

# class globalV:
# 	count = 0
# 	hr = 0.0
# 	br = 0.0
# 	def __init__(self, count):
# 		self.count = count

# port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5)
# gv = globalV(0)
# vts = vitalsign.VitalSign(port)

def Phase_difference(unwarp_phase):
	phase_diff = []
	for tmp in range(len(unwarp_phase)):
		if tmp > 0:
			phase_diff_tmp = unwarp_phase[tmp] - unwarp_phase[tmp - 1]
			phase_diff.append(phase_diff_tmp)
	return phase_diff

def Remove_impulse_noise(phase_diff, thr):
	removed_noise = np.copy(phase_diff)
	for i in range(1, len(phase_diff)-1):
		forward = phase_diff[i] - phase_diff[i-1]
		backward = phase_diff[i] - phase_diff[i+1]
		#print(forward, backward)
		if (forward > thr and backward > thr) or (forward < -thr and backward < -thr):
			removed_noise[i] = phase_diff[i-1] + (phase_diff[i+1] -  phase_diff[i-1])/2
		removed_noise[i] = phase_diff[i]
	return removed_noise

def Amplify_signal(removed_noise):
	for i in range(len(removed_noise)):
		tmp = removed_noise[i]
		if tmp > 0:
			tmp += 1
		elif tmp < 0:
			tmp -= 1
		tmp *= 5
		removed_noise[i] == tmp
	return removed_noise
	
def iir_bandpass_filter_1(data, lowcut, highcut, signal_freq, filter_order, ftype):
    '''
    IIR filter
    '''
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.iirfilter(filter_order, [low, high], rp=5, rs=60, btype='bandpass', ftype = ftype)
    y = signal.lfilter(b, a, data)
    return y

def MLR(data, delta):
    data_s = np.copy(data)
    mean = np.copy(data)
    m = np.copy(data)
    b = np.copy(data)
    for t in range(len(data)):
        if (t - delta ) < 0 or (t + delta + 1) > len(data):
            None
        else:
            start = t - delta
            end = t + delta + 1
            mean[t] = np.mean(data[start:end])

            mtmp = 0
            for i in range(-delta, delta + 1):
                mtmp += i * (data[t + i] - mean[t])
            m[t] = (3 * mtmp) / (delta * (2 * delta + 1) * (delta + 1))
            b[t] = mean[t] - (t * m[t])

    for t in range(len(data)):
        if (t - delta) < 0 or (t + delta + 1) > len(data):
            data_s[t] = data[t]
        else:
            tmp = 0
            for i in range(t - delta, t + delta):
                tmp += m[i] * t + b[i]
            data_s[t] = tmp / (2 * delta + 1)
    return data_s

def feature_detection(data):
	data_v=np.copy(data)
	feature_peak, _ = find_peaks(data)
	feature_valley, _  = find_peaks(-data)
	data_v=np.multiply(np.square(data),np.sign(data))
	return feature_peak,feature_valley,data_v

def feature_compress(feature_peak,feature_valley,time_thr,signal):
	feature_compress_peak=np.empty([1,0])
	feature_compress_valley=np.empty([1,0])
	# sort all the feature
	feature=np.append(feature_peak,feature_valley)

	feature=np.sort(feature)
	# grouping the feature
	ltera=0
	while(ltera < (len(feature)-1)):
		# record start at valley or peak (peak:0 valley:1)
		i, = np.where(feature_peak == feature[ltera])
		if(i.size==0):
			start=1
		else:
			start=0
		ltera_add=ltera
		while(feature[ltera_add+1]-feature[ltera_add]<time_thr):
			# skip the feature which is too close
			ltera_add=ltera_add+1
			#break the loop if it is out of boundary
			if(ltera_add >= (len(feature)-1)):
				break
		# record end at valley or peak (peak:0 valley:1)
		i, = np.where(feature_peak == feature[ltera_add])
		if(i.size==0):
			end=1
		else:
			end=0
		# if it is too close
		if (ltera!=ltera_add):
			# situation1: began with valley end with valley
			if(start==1 and end==1):
				# using the lowest feature as represent
				tmp=(np.min(signal[feature[ltera:ltera_add]]))
				i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera+i])
			#situation2: began with valley end with peak
			elif(start==1 and end==0):
				# using the left feature as valley, right feature as peak
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera])
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera_add])
			#situation3: began with peak end with valley
			elif(start==0 and end==1):
				# using the left feature as peak, right feature as valley
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera])
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera_add])
			#situation4: began with peak end with peak
			elif(start==0 and end==0):
				# using the highest feature as represent
				# tmp=np.array(tmp,dtype = 'float')
				tmp= np.max(signal[feature[ltera:ltera_add]])
				i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera+i])
			ltera=ltera_add
		else:
			# it is normal featur point
			if(start):
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera])
			else:
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera])				
		ltera=ltera+1

	return feature_compress_peak,feature_compress_valley

def candidate_search(signal_v,feature,window_size):
	NT_point=np.empty([1,0])
	NB_point=np.empty([1,0])
	#doing the zero paddding
	signal_pad=np.ones((len(signal_v)+2*window_size))
	signal_pad[window_size:(len(signal_pad)-window_size)]=signal_v
	signal_pad[0:window_size]=signal_v[0]
	signal_pad[(len(signal_pad)-window_size):-1]=signal_v[-1]
	# calaulate the mean and std using windows(for peaks)
	for i in range(len(feature)):
		# for the mean
		window_sum=(np.sum(signal_pad[int(feature[i]):int(feature[i]+2*window_size+1)]))/(window_size*2+1)
		window_var=np.sqrt(np.sum(np.square(signal_pad[int(feature[i]):int(feature[i]+2*window_size+1)]-window_sum))/(window_size*2+1))
		# print('t:',feature[i],'value:',window_sum+window_var)
		#determine if it is NT
		# print(window_var)
		if(signal_v[feature[i].astype(int)]>window_sum and window_var>0.01):
			NT_point=np.append(NT_point,feature[i])
		# determine if it is BT
		elif(signal_v[feature[i].astype(int)]<window_sum and window_var>0.01):
			NB_point=np.append(NB_point,feature[i])

	return NT_point,NB_point

def caculate_breathrate(NT_points,NB_points):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# if both NT and NB are not detected
    if NT_points.shape[0] <= 1 and NB_points.shape[0] <= 1:
        return None
    # if only NT are detected
    elif NT_points.shape[0] > 1 and NB_points.shape[0] <= 1:
        tmp = np.concatenate(([0], NT_points), axis=0)
        tmp_2 = np.concatenate((NT_points, [0]), axis=0)
        aver_NT = tmp_2[1:-1] - tmp[1:-1]
        return 1200 / np.mean(aver_NT)  # (60)*(20)
    # if only NB are detected
    elif NB_points.shape[0] > 1 >= NT_points.shape[0]:
        tmp = np.concatenate(([0], NB_points), axis=0)
        tmp_2 = np.concatenate((NB_points, [0]), axis=0)
        aver_NB = tmp_2[1:-1] - tmp[1:-1]
        return 1200 / np.mean(aver_NB)
    else:
        tmp = np.concatenate(([0], NT_points), axis=0)  # tmp 兩點距離
        tmp_2 = np.concatenate((NT_points, [0]), axis=0)
        aver_NT = tmp_2[1:-1] - tmp[1:-1]
        tmp = np.concatenate(([0], NB_points), axis=0)
        tmp_2 = np.concatenate((NB_points, [0]), axis=0)
        aver_NB = tmp_2[1:-1] - tmp[1:-1]
        aver = (np.mean(aver_NB) + np.mean(aver_NT)) / 2
    return 1200 / aver    #因為一個周期是20Hz所以時間就是1/20，然後是算每分鐘，所以還要再除以1/60

def detect_Breath(unw_phase, a): #,lowHz
	replace = False
	N = 0
	T = 0
	# Phase difference
	phase_diff = Phase_difference(unw_phase)

	# RemoveImpulseNoise
	re_phase_diff = Remove_impulse_noise(phase_diff, int(a[0]))

	# Linear amplify
	amp_sig = Amplify_signal(re_phase_diff)

	# Bandpass signal (cheby2)
	bandpass_sig = iir_bandpass_filter_1(amp_sig, float(a[1]), float(a[2]), int(a[3]), int(a[4]), "cheby2") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
	N = len(bandpass_sig)
	T = 1 / 20
	bps_fft = fft(bandpass_sig)
	bps_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)
	index_of_fftmax = np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2)
	if index_of_fftmax < 1.17:
		replace = True

	# Smoothing signal
	smoothing_signal = MLR(bandpass_sig, int(a[5]))  # Breath = 9, Heart = 6, Delta = 1

	#detect the feature
	feature_peak, feature_valley, feature_sig = feature_detection(smoothing_signal) #找出所有的波峰及波谷

	#compress with window size 7
	compress_peak, compress_valley = feature_compress(feature_peak, feature_valley, int(a[6]), smoothing_signal)  # Br: 20 Hr: 6  ex: 25

	# Feature sort
	compress_feature = np.append(compress_peak, compress_valley)
	compress_feature = np.sort(compress_feature)

	# Candidate_search
	NT_points, NB_points = candidate_search(smoothing_signal, compress_feature, int(a[7]))  # breath = 18 hreat = 4 ex: 7

	rate = caculate_breathrate(NT_points, NB_points)
	return rate, replace, index_of_fftmax

# mmWave toolbox
# def uartGetTLVdata():
# 	br_array = deque(maxlen=300)
# 	for i in range(300):
# 		br_array.append(0)

# 	port.flushInput()
# 	while True:
# 		(dck , vd, rangeBuf) = vts.tlvRead(False)
# 		vs = vts.getHeader()
# 		if dck:
# 			#print("unwrapPhasePeak_mm:{0:.4f}".format(vd.unwrapPhasePeak_mm))
# 			br_array.append(vd.unwrapPhasePeak_mm)
# 			print(detect_Breath(br_array))

def timing(sec):
	print(f"Get ready in {sec} seconds...")
	current_time = 0
	print(f"{current_time} sec")
	while current_time != sec:
		time.sleep(1)
		current_time += 1
		print(f"{current_time} sec")

if __name__ == "__main__":
	# Timing before start
	timing(4)
	
	# Connect the radar
	port_read = serial.Serial("COM3", baudrate=921600, timeout=0.5)
	vital_sig = vitalsign_v2.VitalSign(port_read)

	# Initialization time
	time_Start = time.time()
	port_read.flushInput()

	# Start recording
	raw_sig = []  # 訊號的窗格
	energe_br = []  # 呼吸能量的窗格
	energe_hr = []  # 心跳能量的窗格
	heart_ti = []
	coco = True  # 時間開關
	a = [[1.5, 0.125, 0.55, 20, 5, 2, 22, 17], [1.5, 0.9, 1.9, 20, 9, 2, 5, 4]]
	a = np.array(a)
	print("Recoding data...")
	while True:
		(state, vsdata, rangeBuf) = vital_sig.tlv_read(False)
		vs = vital_sig.getHeader()
		if state:
			ct = datetime.datetime.now()
			raw_sig.append(vsdata.unwrapPhasePeak_mm)
			energe_br.append(vsdata.sumEnergyBreathWfm)
			energe_hr.append(vsdata.sumEnergyHeartWfm)
			heart_ti.append(vsdata.rsv[1])
			time_End = time.time()
			if coco:
				print(f"Elapsed time (sec): {time_End - time_Start}")
			if len(raw_sig) >= 40 * 20:
				try:
					current_window_sig = raw_sig[-40*20:]
					current_window_ebr = energe_br[-3*20:]
					current_window_ehr = energe_hr[-3*20:]
					current_heart_ti = heart_ti[-40*20:]
					if time_End - time_Start >= 1:
						coco = False
						time_Start = time_End

						# 判別有沒有呼吸與有沒有人 (憋氣可以判別)
						if np.mean(current_window_ebr) > 50000000 and np.mean(current_window_ehr) > 50:
							br_rate, replace1 ,index_of_fftmax = detect_Breath(current_window_sig, a[0][:])
							hr_rate, replace1 ,index_of_fftmax = detect_Breath(current_window_sig, a[1][:])
							if replace1:
								result_rate = int(np.mean(np.array(current_heart_ti))) 
							br_rpm = np.round(br_rate)
							hr_rpm = np.round(hr_rate)
							print(f"Breathe Rate per minute: {br_rpm}")
							print(f"Heart Rate per minute: {hr_rpm}")					
						elif np.mean(current_window_ebr) < 50000000 and np.mean(current_window_ehr) < 50:
							print("No People")
				# Ctrl C 中斷
				except KeyboardInterrupt:
					print("Interrupt")

	# ground_truth_txt='./dataset/a1.txt'
	# ground_truth = np.loadtxt(ground_truth_txt)
	# for i in range(0,600,300):
	# 	points=detect_Breath(ground_truth[0+ i:300+ i])
	# 	print(points)
	# uartGetTLVdata()


