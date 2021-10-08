"""
最一開始的程式，都沒動過
"""
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
def Phase_difference(unwarp_phase):
	phase_diff = np.zeros((len(unwarp_phase),))
	for i in range(len(unwarp_phase)-1):
		phase_diff[i] = unwarp_phase[i+1] - unwarp_phase[i]
	return phase_diff

def Remove_impulse_noise(phase_diff, thr):
	removed_noise = np.zeros((len(phase_diff),))
	for i in range(1, len(phase_diff)-1):
		forward = phase_diff[i] - phase_diff[i-1]
		backward = phase_diff[i] - phase_diff[i+1]
		#print(forward, backward)
		if (forward > thr and backward > thr) or (forward < -thr and backward < -thr):
			removed_noise[i] = phase_diff[i-1] + (phase_diff[i+1] -  phase_diff[i-1])/2
		removed_noise[i] = phase_diff[i]
	return removed_noise

def Amplify_signal(removed_noise):
	return removed_noise*1.0

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def normolize(data):
    output=(data-np.min(data))/(np.max(data)-np.min(data))
    return output

def MLR(data,delta):
	data_s=np.copy(data)
	mean=np.copy(data)
	m=np.copy(data)
	b=np.copy(data)
	#calculate m
	for t in range(len(data)):
		# constraint
		if ((t-delta)<0 or (t+delta)>(len(data)-1)):
			None
		# if the sliding window is in the boundary
		else:
			mean[t]=(np.sum(data[int(t-delta):int(t+delta+1)]))/(2*delta+1)
			# calaulate the sigma
			mtmp=0
			for j in range(-delta,delta+1):
				mtmp=mtmp+(j*(data[j+t]-mean[t]))
			m[t] = 3*mtmp/(delta*(2*delta+1)*(delta+1))
			b[t] = mean[t]-(t*m[t])

	for t in range(len(data)):
		# constraint
		# if the sliding window is in the boundary
		if ((t-2*delta)>=0 and (t+2*delta)<=(len(data)-1)):
			# calaulate smooth ECG
			tmp=0
			for i in range(-delta,delta+1):
				tmp=tmp+(t*m[t+i]+b[t+i])
				# print(i)
			data_s[t]=tmp/(2*delta+1)
		else:
			data_s[t]=data[t]
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
	if(NT_points.shape[0]<=1 and NB_points.shape[0]<=1):
		return None
	# if only NT are detected
	elif(NT_points.shape[0]>1 and NB_points.shape[0]<=1):
		tmp=np.concatenate(([0],NT_points),axis=0)
		tmp_2=np.concatenate((NT_points,[0]),axis=0)
		aver_NT=tmp_2[1:-1]-tmp[1:-1]
		return 1200/np.mean(aver_NT) #(60)*(20)
	# if only NB are detected
	elif(NB_points.shape[0]>1 and NT_points.shape[0]<=1):
		tmp=np.concatenate(([0],NB_points),axis=0)
		tmp_2=np.concatenate((NB_points,[0]),axis=0)
		aver_NB=tmp_2[1:-1]-tmp[1:-1]
		return 1200/np.mean(aver_NB)
	else:
		tmp=np.concatenate(([0],NT_points),axis=0)   #tmp 兩點距離
		tmp_2=np.concatenate((NT_points,[0]),axis=0)
		aver_NT=tmp_2[1:-1]-tmp[1:-1]
		tmp=np.concatenate(([0],NB_points),axis=0)
		tmp_2=np.concatenate((NB_points,[0]),axis=0)
		aver_NB=tmp_2[1:-1]-tmp[1:-1]
		aver=(np.mean(aver_NB)+np.mean(aver_NT))/2
	return 1200/aver    #因為一個周期是20Hz所以時間就是1/20，然後是算每分鐘，所以還要再除以1/60

def detect_Breath(data): #,lowHz

	data_txt = Phase_difference(data)

	data_txt = Remove_impulse_noise(data_txt, 2)

	data_txt = Amplify_signal(data_txt)

	data_txt = butter_bandpass_filter(data_txt, 0.1, 0.33, 20, order=5)  #breath = 0.1~0.33~0.35, order = 4

	signal_c=MLR(data_txt,1) #平滑化  #breath = 9 hreat= 6

	#detect the feature
	feature_peak,feature_valley,signal_v=feature_detection(signal_c) #找出所有的波峰及波谷

	#compress with window size 7
	feature_peak,feature_valley=feature_compress(feature_peak,feature_valley,7,signal_c) #將多餘的峰跟谷捨去 #breath= 14 hreat = 6

	feature=np.append(feature_peak,feature_valley)
	feature=np.sort(feature)

	NT_points,NB_points=candidate_search(signal_c,feature,7) #breath = 18 hreat = 4

	rate=caculate_breathrate(NT_points,NB_points)

	return rate 
def detect_Heart(data): #,lowHz
	
	data_txt = Phase_difference(data)

	data_txt = Remove_impulse_noise(data_txt, 0.15)

	data_txt = Amplify_signal(data_txt)

	data_txt = butter_bandpass_filter(data_txt, 0.8, 1.32, 20, order=5)  #breath = 0.1~0.33~0.35, order = 4

	signal_c=MLR(data_txt,1) #平滑化  #breath = 9 hreat= 6

	#detect the feature
	feature_peak,feature_valley,signal_v=feature_detection(signal_c) #找出所有的波峰及波谷

	#compress with window size 7
	feature_peak,feature_valley=feature_compress(feature_peak,feature_valley,7,signal_c) #將多餘的峰跟谷捨去 #breath= 14 hreat = 6

	feature=np.append(feature_peak,feature_valley)
	feature=np.sort(feature)

	NT_points,NB_points=candidate_search(signal_c,feature,7) #breath = 18 hreat = 4

	rate=caculate_breathrate(NT_points,NB_points)

	return rate 

if __name__ == "__main__":
	ground_truth_txt='./dataset/c.txt'
	ground_truth = np.loadtxt(ground_truth_txt)
	'''
	for i in range(0,600,300):
		points=detect_Breath(ground_truth[0+ i:300+ i])
		print(points)
		points=detect_Heart(ground_truth[0+ i:300+ i])
		print(points)
	'''
	points=detect_Breath(ground_truth)
	print(points)
	points=detect_Heart(ground_truth)
	print(points)