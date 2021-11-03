import numpy as np
from numpy.lib.function_base import unwrap
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
from scipy import signal
from scipy.fftpack import fft
import seaborn as sns
from tqdm import tqdm
from number_analyze import breath_analyze
from losscal import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle #pickle模組
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
sns.set()

def calculate_l1_loss(gt, pr):
    temp = 0
    for i in range(len(gt)):
        temp += abs(gt[i] - pr[i])
    return temp/len(gt)

def filter_RemoveImpulseNoise(phase_diff, dataPrev2, dataPrev1, dataCurr, thresh):
    pDataIn = [0, 0, 0]
    pDataIn[0] = float(phase_diff[dataPrev2])
    pDataIn[1] = float(phase_diff[dataPrev1])
    pDataIn[2] = float(phase_diff[dataCurr])

    backwardDiff = pDataIn[1] - pDataIn[0]
    forwardDiff  = pDataIn[1] - pDataIn[2]

    x1 = 0
    x2 = 2
    y1 = pDataIn[0]
    y2 = pDataIn[2]
    x  = 1
    thresh = float(thresh)

    if ((forwardDiff > thresh) and (backwardDiff > thresh)) or ((forwardDiff < -thresh) and (backwardDiff < -thresh)):
        y = y1 + (((x - x1) * (y2 - y1)) / (x2 - x1))
    else:
        y = pDataIn[1]
    return y

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def iir_bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    '''
    IIR filter
    '''
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.iirfilter(filter_order, [low, high], btype='bandpass', ftype='bessel')
    y = signal.lfilter(b, a, data)
    return y

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


def firwin_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    numtaps = 29
    fir_coeff = signal.firwin(numtaps, high)
    filtered_signal = signal.lfilter(fir_coeff, 1.0, data)
    return filtered_signal

def lowpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    b, a = signal.butter(filter_order, low, btype='low')
    y = signal.filtfilt(b, a, data)
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

def feature_detection(smoothing_signal):
    feature_peak, _ = signal.find_peaks(smoothing_signal)
    feature_valley, _ = signal.find_peaks(-smoothing_signal)
    data_value = np.multiply(np.square(smoothing_signal), np.sign(smoothing_signal))
    return feature_peak, feature_valley, data_value

def feature_compress(feature_peak, feature_valley, time_thr, signal):
    feature_compress_peak = np.empty([1, 0])
    feature_compress_valley = np.empty([1, 0])
    # Sort all the feature
    feature = np.append(feature_peak, feature_valley)
    feature = np.sort(feature)

    # Grouping the feature
    ltera = 0
    while ltera < (len(feature) - 1):
        # Record start at valley or peak (peak:0 valley:1)
        i, = np.where(feature_peak == feature[ltera])
        if i.size == 0:
            start = 1
        else:
            start = 0
        ltera_add = ltera
        while feature[ltera_add + 1] - feature[ltera_add] < time_thr:
            # skip the feature which is too close
            ltera_add = ltera_add + 1
            # break the loop if it is out of boundary
            if ltera_add >= (len(feature) - 1):
                break
        # record end at valley or peak (peak:0 valley:1)
        i, = np.where(feature_peak == feature[ltera_add])
        if i.size == 0:
            end = 1
        else:
            end = 0
        # if it is too close
        if ltera != ltera_add:
            # situation1: began with valley end with valley
            if start == 1 and end == 1:
                # using the lowest feature as represent
                tmp = (np.min(signal[feature[ltera:ltera_add]]))
                i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
                feature_compress_valley = np.append(feature_compress_valley, feature[ltera + i])
            # situation2: began with valley end with peak
            elif start == 1 and end == 0:
                # using the left feature as valley, right feature as peak
                feature_compress_valley = np.append(feature_compress_valley, feature[ltera])
                feature_compress_peak = np.append(feature_compress_peak, feature[ltera_add])
            # situation3: began with peak end with valley
            elif start == 0 and end == 1:
                # using the left feature as peak, right feature as valley
                feature_compress_peak = np.append(feature_compress_peak, feature[ltera])
                feature_compress_valley = np.append(feature_compress_valley, feature[ltera_add])
            # situation4: began with peak end with peak
            elif start == 0 and end == 0:
                # using the highest feature as represent
                # tmp=np.array(tmp,dtype = 'float')
                tmp = np.max(signal[feature[ltera:ltera_add]])
                i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
                feature_compress_peak = np.append(feature_compress_peak, feature[ltera + i])
            ltera = ltera_add
        else:
            # it is normal feature point
            if start:
                feature_compress_valley = np.append(feature_compress_valley, feature[ltera])
            else:
                feature_compress_peak = np.append(feature_compress_peak, feature[ltera])
        ltera = ltera + 1

    # # Last one need to be save
    # if feature[len(feature) - 1] in feature_peak:
    #     feature_compress_peak = np.append(feature_compress_peak, feature[len(feature) - 1])
    # elif feature[len(feature) - 1] in feature_valley:
    #     feature_compress_valley = np.append(feature_compress_valley, feature[len(feature) - 1])

    return feature_compress_peak.astype(int), feature_compress_valley.astype(int)

def candidate_search(signal, feature, window_size):
    NT_point = np.empty([1, 0])
    NB_point = np.empty([1, 0])
    signal_pad = np.ones((len(signal) + 2 * window_size))
    signal_pad[window_size:(len(signal_pad) - window_size)] = signal
    signal_pad[0:window_size] = signal[0]
    signal_pad[(len(signal_pad) - window_size):-1] = signal[-1]
    # calaulate the mean and std using windows(for peaks)
    for i in range(len(feature)):
        # Calculate the mean
        window_mean = (np.sum(signal_pad[int(feature[i]):int(feature[i] + 2 * window_size + 1)])) / (
                    window_size * 2 + 1)
        # Calculate the std
        window_std = np.sqrt(
            np.sum(np.square(signal_pad[int(feature[i]):int(feature[i] + 2 * window_size + 1)] - window_mean)) / (
                    window_size * 2 + 1))
        # Determine if it is NT
        # 跟paper不同
        # if signal_v[feature[i].astype(int)] > window_mean + window_std:
        if signal[feature[i].astype(int)] > window_mean and window_std > 0.01:
            NT_point = np.append(NT_point, feature[i])
        # Determine if it is BT
        # elif signal_v[feature[i].astype(int)] < window_mean - window_std:
        elif signal[feature[i].astype(int)] < window_mean and window_std > 0.01:
            NB_point = np.append(NB_point, feature[i])
    return NT_point.astype(int), NB_point.astype(int)

def caculate_breathrate(NT_points, NB_points):
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
    return 1200 / aver

def detect_breath(unw_phase, count, disp):
    replace = False
    # Unwrap phase
    raw = unw_phase

    # Phase difference
    phase_diff = []
    for tmp in range(len(unw_phase)):
        if tmp > 0:
            phase_diff_tmp = unw_phase[tmp] - unw_phase[tmp - 1]
            phase_diff.append(phase_diff_tmp)

    # RemoveImpulseNoise
    new_phase_diff = np.copy(phase_diff)
    for i in range(1, int(len(phase_diff))-1):
        dataPrev2 = i - 1
        dataPrev1 = i
        dataCurr = i + 1
        
        a = filter_RemoveImpulseNoise(phase_diff, dataPrev2, dataPrev1, dataCurr, 1.5)
        if a > 0:
            a += 1
        elif a < 0:
            a -= 1
        a *= 5
        new_phase_diff[i] = a
    # -------------- Removed noise -------------- 
    removed_noise = 0
    for i in range(len(phase_diff)):
        removed_noise += phase_diff[i] - new_phase_diff[i]
    # print(f'Sum of remove impulse noise: {removed_noise}')

    #butter ellip cheby2
    bandpass_sig = iir_bandpass_filter_1(new_phase_diff, 0.1125, 0.55, 20, 5, "cheby2") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
    #bandpass_sig = butter_bandpass_filter(new_phase_diff, 0.8, 2, 20, 5) # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
    #bandpass_sig = iir_bandpass_filter_1(bandpass_sig, 0.8, 2, 20, 5, "cheby1") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
    #bandpass_sig = firwin_filter(new_phase_diff, 0.8, 2, 20, 5)
    #bandpass_sig = lowpass_filter(bandpass_sig, 2, 2, 20, 5)
    N = len(bandpass_sig)
    T = 1 / 20
    bps_fft = fft(bandpass_sig)
    bps_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)   
    #print(np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2))
    index_of_fftmax = np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2)
    print(index_of_fftmax)
    if index_of_fftmax < 0.215:
        replace = True
    # Smoothing signal
    smoothing_signal = MLR(bandpass_sig, 2)  # Breath = 9, Heart = 6, Delta = 1

    # Try to make smoothing values (Sv) (Sv > 1 or Sv < -1)
    # smoothing_signal = np.copy(smoothing_signal)
    # for i in range(1, int(len(smoothing_signal))-1):
    #     if smoothing_signal[i] > 0:
    #         tmp_s = smoothing_signal[i] + 1
    #         smoothing_signal[i] = tmp_s
    #     elif smoothing_signal[i] < 0:
    #         tmp_s = smoothing_signal[i] - 1
    #         smoothing_signal[i] = tmp_s

    # Feature detect
    feature_peak, feature_valley, feature_sig = feature_detection(smoothing_signal)

    # Feature compress
    compress_peak, compress_valley = feature_compress(feature_peak, feature_valley, 22, smoothing_signal)  # Br: 20 Hr: 6  ex: 25

    # Feature sort
    compress_feature = np.append(compress_peak, compress_valley)
    compress_feature = np.sort(compress_feature)

    # Candidate_search
    NT_points, NB_points = candidate_search(smoothing_signal, compress_feature, 17)  # breath = 18 hreat = 4 ex7

    # Breath rate
    rate = caculate_breathrate(NT_points, NB_points)
    print(f'Rate: {rate}')

    if disp:
        # Define 
        sampling_rate = 20
        record_time = len(unw_phase) / sampling_rate

        # Unwrap phase
        plt.figure()
        raw_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(raw))
        plt.plot(raw_x, raw)
        plt.title('Unwrap phase')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # Phase difference
        plt.figure()
        phase_diff_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(phase_diff))
        plt.plot(phase_diff_x, phase_diff, label="$sin(x)$")
        plt.title('Phase difference')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # RemoveImpulseNoise
        plt.figure()
        new_phase_diff_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(new_phase_diff))
        plt.plot(new_phase_diff_x, new_phase_diff, label="$sin(x)$", color='b')
        plt.title('Remove Impulse Noise')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # Bandpass signal (Butter worth)
        plt.figure()
        bandpass_sig_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(bandpass_sig))
        plt.plot(bandpass_sig_x, bandpass_sig)
        plt.title('Bandpass signal')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # Smoothing signal
        plt.figure()
        smoothing_signal_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(smoothing_signal))
        plt.plot(smoothing_signal_x, smoothing_signal)
        plt.title('Smoothing signal')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')


        # Feature detect
        plt.figure()
        feature_peak_x = (record_time * count) + feature_peak/len(feature_sig) * record_time
        feature_valley_x = (record_time * count) + feature_valley/len(feature_sig) * record_time
        feature_sig_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(feature_sig))
        plt.plot(feature_sig_x, feature_sig)
        plt.plot(feature_peak_x, feature_sig[feature_peak], 'bo')
        plt.plot(feature_valley_x, feature_sig[feature_valley], 'ro')
        plt.title('Feature detect')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # Feature compress
        plt.figure()
        compress_peak_x = (record_time * count) + compress_peak/len(feature_sig) * record_time
        compress_valley_x = (record_time * count) + compress_valley/len(feature_sig) * record_time
        feature_sig_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(feature_sig))
        plt.plot(feature_sig_x, feature_sig)
        plt.plot(compress_peak_x, feature_sig[compress_peak], 'bo')
        plt.plot(compress_valley_x, feature_sig[compress_valley], 'ro')
        plt.title('Feature compress')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # Candidate_search
        plt.figure()
        candi_peak_x = (record_time * count) + NT_points/len(smoothing_signal) * record_time
        candi_valley_x = (record_time * count) + NB_points/len(smoothing_signal) * record_time
        candidate_search_x = np.linspace(0 + (record_time * count), record_time + (record_time * count), len(smoothing_signal))
        plt.plot(candidate_search_x, smoothing_signal)
        plt.plot(candi_peak_x, smoothing_signal[NT_points], 'bo')
        plt.plot(candi_valley_x, smoothing_signal[NB_points], 'ro')
        plt.title('Candidate_search')
        plt.xlabel('Time (sec)')
        plt.ylabel('Phase (radians)')

        # ----------------------
        # FFT (Before and After)
        plt.figure()
        # Before bandpass
        N = len(new_phase_diff)
        T = 1 / sampling_rate
        ori_fft = fft(new_phase_diff)
        ori_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)
        # plt.subplot(2, 1, 1)
        plt.plot(ori_fft_x, 2 / N * np.abs(ori_fft[:N // 2]))
        # plt.legend(labels=['Phase diff FFT'], loc='upper right')
        plt.title('Phase diff FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Ampitude')

        # After bandpass
        plt.figure()
        N = len(bandpass_sig)
        T = 1 / sampling_rate
        bps_fft = fft(bandpass_sig)
        bps_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)
        # plt.subplot(2, 1, 2)
        # plt.legend(labels=['Bandpassed FFT'], loc='upper right')
        plt.title('Bandpassed FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Ampitude')
        plt.plot(bps_fft_x, 2 / N * np.abs(bps_fft[:N // 2]))
        print(np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2))
        plt.show()

    return rate, replace, index_of_fftmax

def plot_scatter(
            all_index_of_fftmax,
            all_gt_array, 
            all_confidenceMetricBreathOut_std,
            all_confidenceMetricBreathOut_xCorr_std,
            all_confidenceMetricBreathOut_mean,
            all_confidenceMetricBreathOut_xCorr_mean,
            all_breathingRateEst_FFT_std,all_breathingRateEst_FFT_mean,
            all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
            all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean, 
            all_sumEnergyBreathWfm_mean, 
            all_sumEnergyBreathWfm_std, 
            all_sumEnergyHeartWfm_mean, 
            all_sumEnergyHeartWfm_std):
    plt.xlabel('all_breathingEst_xCorr_mean')
    plt.ylabel('heartrate_groundtruth')
    plt.scatter(all_breathingEst_xCorr_mean, all_gt_array)
    plt.show()
#all_breathingRateEst_FFT_mean, all_breathingEst_xCorr_mean
def knn_test(
            predict_array,
            all_index_of_fftmax,
            all_gt_array, 
            all_confidenceMetricBreathOut_std,
            all_confidenceMetricBreathOut_xCorr_std,
            all_confidenceMetricBreathOut_mean,
            all_confidenceMetricBreathOut_xCorr_mean,
            all_breathingRateEst_FFT_std,all_breathingRateEst_FFT_mean,
            all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
            all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean, 
            all_sumEnergyBreathWfm_mean, 
            all_sumEnergyBreathWfm_std, 
            all_sumEnergyHeartWfm_mean, 
            all_sumEnergyHeartWfm_std,
            test_array1, test_array2, all_ti_og_br):
    find_best = False

    all_data = [
            all_index_of_fftmax,
            all_confidenceMetricBreathOut_std,
            all_confidenceMetricBreathOut_xCorr_std,
            all_confidenceMetricBreathOut_mean,
            all_confidenceMetricBreathOut_xCorr_mean,
            all_breathingRateEst_FFT_std,all_breathingRateEst_FFT_mean,
            all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
            all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean, 
            all_sumEnergyBreathWfm_mean, 
            all_sumEnergyBreathWfm_std, 
            all_sumEnergyHeartWfm_mean, 
            all_sumEnergyHeartWfm_std, all_ti_og_br]
    all_data = [all_index_of_fftmax, all_breathingEst_xCorr_mean, all_breathingRateEst_FFT_mean]
    all_data = np.array(all_data).transpose()
    all_data = preprocessing.scale(all_data)

    label_knn = []
    for index, i in enumerate(all_gt_array): #做ml分類的label
        if all_gt_array[index] <= 15:
            label_knn.append(1)
        else:
            label_knn.append(0)
    label_knn = np.array(label_knn)
    test_array1 = np.array(test_array1)
    test_array2 = np.array(test_array2)
    all_gt_array = np.array(all_gt_array)
    all_ti_og_br = np.array(all_ti_og_br)
    kf = StratifiedKFold(n_splits = 3, random_state = 69, shuffle = True)
    knn_p = []
    svm_p = []
    rf_p = []
    ti_replace_result_kf_number =  [] #算K-FOLD用的
    og_result_kf_number = []
    for train_index, test_index in kf.split(all_data, label_knn):

        X_train, X_test = all_data[train_index], all_data[test_index]
        y_train, y_test = label_knn[train_index], label_knn[test_index]
        og_result_kf = test_array1[test_index] #最原始的算法輸出
        ti_replace_result_kf = test_array2[test_index] #TI取代的算法輸出
        gt_kf = all_gt_array[test_index]  #GT
        all_ti_og_kf = all_ti_og_br[test_index] #TI 原始輸出

        #print("TI Tr取代：", calculate_l1_loss(gt_kf, ti_replace_result_kf))   
        ti_replace_result_kf_number.append(calculate_l1_loss(gt_kf, ti_replace_result_kf))    

        neigh = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
        neigh.fit(X_train, y_train)
        knn_p.append(neigh.score(X_test,y_test))

        clf_rbf = svm.SVC(kernel="rbf", C = 5, gamma = 0.01, random_state = 69)
        clf_rbf.fit(X_train, y_train) #用TRAIN資料下去訓練
        svm_p.append(clf_rbf.score(X_test,y_test))

        rf = RandomForestRegressor(n_estimators = 20, random_state = 69)
        rf.fit(X_train, y_train)
        rf_p.append(rf.score(X_test,y_test))

        y_test_pre_rbf = clf_rbf.predict(X_test) #用TEST下去預測
        for index, x in enumerate(y_test_pre_rbf):
            if x == 1: #如果SVM輸出1代表心跳小於70
                og_result_kf[index] = all_ti_og_kf[index] #就用TI原始輸出替代最原始的算法輸出
                
        #print("TI SVM取代：", calculate_l1_loss(gt_kf, og_result_kf))   
        og_result_kf_number.append(calculate_l1_loss(gt_kf, og_result_kf))     

    print("AVG knn 分類表現 ", np.mean(np.array(knn_p)))
    print("AVG svm 分類表現 ", np.mean(np.array(svm_p)))
    print("AVG rf 分類表現 ", np.mean(np.array(rf_p)))
    print("AVG TI Tr取代：", np.mean(np.array(ti_replace_result_kf_number)))
    print("AVG TI SVM取代：", np.mean(np.array(og_result_kf_number)))

    clf_rbf = svm.SVC(kernel="rbf", C = 5, gamma = 0.01, random_state = 69)
    clf_rbf.fit(all_data, label_knn)   
    with open('save/svm_br.pickle', 'wb') as f:
        pickle.dump(clf_rbf, f)
    if find_best:
        parameters = {'gamma': [0.001, 0.01, 0.1, 1, 5, 10], 'C':[0.001, 0.01, 0.1, 1, 5, 10], 'degree':[3, 4, 5], 'kernel': ["rbf", "linear"]}
        #parameters = {'kernel': ["rbf", "poly", "linear"]}
        #n_jobs =-1使用全部CPU并行多线程搜索
        gs = GridSearchCV(svm.SVC(), parameters, refit = True, cv = 3, verbose = 1, n_jobs = -1)
        gs.fit(all_data, label_knn) #Run fit with all sets of parameters.
        print('最优参数: ',gs.best_params_)
        print('最佳性能: ', gs.best_score_)

if __name__ == '__main__':

    # Initial setting
    count = 0
    count_all = 0
    absolute_error = 0
    disp = False
    diagram_disp = False  # <新增> 是否顯示圖表
    scatter_disp = False
    knn = True
    all_pr_array = []
    all_gt_array = []
    all_ti_og_br = []
    all_ti_og_hr = []
    all_index_of_fftmax = []

    all_std_of_phase_diff = []
    all_breathingRateEst_FFT_std = []
    all_breathingRateEst_FFT_mean = []
    all_breathingEst_xCorr_std = []
    all_breathingEst_xCorr_mean = []
    all_breathingEst_peakCount_std = []
    all_breathingEst_peakCount_mean = []

    all_confidenceMetricBreathOut_std = []
    all_confidenceMetricBreathOut_xCorr_std = []
    all_confidenceMetricBreathOut_mean = []
    all_confidenceMetricBreathOut_xCorr_mean = []
    all_sumEnergyBreathWfm_mean = []
    all_sumEnergyBreathWfm_std = []
    all_sumEnergyHeartWfm_mean = []
    all_sumEnergyHeartWfm_std = []
    test_array1 = []#原始算法輸出
    test_array2 = []#ti輸出值輸出(Tr)
    sample_total = 0
    acc_sample_total = 0
    for user in tqdm(os.listdir("dataset")):
        if os.path.isdir(os.path.join("dataset", user, "gt_br")):
            predict_array = []
            ground_truth_array = []
            ti_predict_array = []
            files_path = os.path.join("dataset", user, "0.8")
            ground_truth_files_path = os.path.join("dataset", user, "gt_br")
            files = os.listdir(files_path)        

            for  name  in os.listdir(ground_truth_files_path):
                with open(os.path.join(ground_truth_files_path, name)) as f:
                    for line in f.readlines():
                        ground_truth_array.append(int(line))
                        all_gt_array.append(int(line))
            for tmp in range(0, len(files)//2, 1):
                file = files[tmp]
                print(f'\nCurrent file: {file}')
                datas_path = os.path.join(files_path, file)
                vitial_sig = pd.read_csv(datas_path)
                unwrapPhase = vitial_sig['unwrapPhasePeak_mm'].values
                heart = vitial_sig['rsv[1]'].values
                breath = vitial_sig['rsv[0]'].values

                confidenceMetricBreathOut_std = np.std(vitial_sig['confidenceMetricBreathOut'].values)
                confidenceMetricBreathOut_xCorr_std = np.std(vitial_sig['confidenceMetricBreathOut_xCorr'].values)
                confidenceMetricBreathOut_mean = np.mean(vitial_sig['confidenceMetricBreathOut'].values)
                confidenceMetricBreathOut_xCorr_mean = np.mean(vitial_sig['confidenceMetricBreathOut_xCorr'].values)
                breathingRateEst_FFT_std = np.std(vitial_sig['breathingRateEst_FFT'].values)
                breathingRateEst_FFT_mean = np.mean(vitial_sig['breathingRateEst_FFT'].values)
                breathingEst_xCorr_std = np.std(vitial_sig['breathingEst_xCorr'].values)
                breathingEst_xCorr_mean = np.mean(vitial_sig['breathingEst_xCorr'].values)
                breathingEst_peakCount_std = np.std(vitial_sig['breathingEst_peakCount'].values)
                breathingEst_peakCount_mean = np.mean(vitial_sig['breathingEst_peakCount'].values)
                sumEnergyBreathWfm_mean = np.mean(vitial_sig['sumEnergyBreathWfm'].values)
                sumEnergyBreathWfm_std = np.std(vitial_sig['sumEnergyBreathWfm'].values)
                sumEnergyHeartWfm_mean = np.mean(vitial_sig['sumEnergyHeartWfm'].values)
                sumEnergyHeartWfm_std = np.std(vitial_sig['sumEnergyHeartWfm'].values)

                all_ti_og_br.append(int(np.mean(breath)))
                ti_predict_array.append(int(np.mean(breath)))
                sample_total += 1
                for i in range (0, 800, 800):  # 0, 600, 1200
                    result_rate, replace1, index_of_fftmax = detect_breath(unwrapPhase[0 + i: 800 + i], count, disp)
                    test_array1.append(round(result_rate))
                    if replace1:
                        result_rate = int(np.mean(breath))
                    all_index_of_fftmax.append(index_of_fftmax)  
                    test_array2.append(round(result_rate))
                    all_confidenceMetricBreathOut_std.append(confidenceMetricBreathOut_std)
                    all_confidenceMetricBreathOut_xCorr_std.append(confidenceMetricBreathOut_xCorr_std)
                    all_confidenceMetricBreathOut_mean.append(confidenceMetricBreathOut_mean)
                    all_confidenceMetricBreathOut_xCorr_mean.append(confidenceMetricBreathOut_xCorr_mean)
                    all_breathingRateEst_FFT_std.append(breathingRateEst_FFT_std)
                    all_breathingRateEst_FFT_mean.append(breathingRateEst_FFT_mean)
                    all_breathingEst_xCorr_std.append(breathingEst_xCorr_std)
                    all_breathingEst_xCorr_mean.append(breathingEst_xCorr_mean)
                    all_breathingEst_peakCount_std.append(breathingEst_peakCount_std)
                    all_breathingEst_peakCount_mean.append(breathingEst_peakCount_mean)
                    all_sumEnergyBreathWfm_mean.append(sumEnergyBreathWfm_mean)
                    all_sumEnergyBreathWfm_std.append(sumEnergyBreathWfm_std)
                    all_sumEnergyHeartWfm_mean.append(sumEnergyHeartWfm_mean)
                    all_sumEnergyHeartWfm_std.append(sumEnergyHeartWfm_std)

                    predict_array.append(round(result_rate))
                    all_pr_array.append(round(result_rate))
                    if result_rate != None:
                        absolute_error = absolute_error + abs(16 - result_rate)
                        count_all += 1
                    else:
                        print('\nEnding')
                    count += 1
                count = 0
                    
            print(user)
            print("predict_array")
            print(predict_array)
            print("ti_predict_array")
            print(ti_predict_array)
            print("ground_truth_array")    
            print(ground_truth_array)
            print("L1 lOSS",calculate_l1_loss(ground_truth_array, predict_array))
            for i in range(len(ground_truth_array)):
                if np.abs(ground_truth_array[i] - predict_array[i]) <= 2:
                    acc_sample_total+=1            
            print("TI L1 lOSS",calculate_l1_loss(ground_truth_array, ti_predict_array))
    print("---------------------------------------------------------------------------")
    print("AVG L1 lOSS",calculate_l1_loss(all_gt_array, all_pr_array))
    print("AVG TI  L1 lOSS",calculate_l1_loss(all_gt_array, all_ti_og_br))
    print("Total sample：", sample_total)
    print("Total acc sample：", acc_sample_total)
    print("------------------------------Ours----------------------------------------")
    ar = breath_analyze(all_pr_array, all_gt_array)
    print("------------------------------TI------------------------------------------")
    ar2 = breath_analyze(all_ti_og_br, all_gt_array)

    if diagram_disp:
        # loss diagram
        diagram(ar, ar2, current_type='b')    # current_type設定要畫哪種圖: 'h' = heart, 'b' = breath 

        # 資料分布
        data_distribution(all_pr_array, all_ti_og_br, all_gt_array, current_type='b')  # current_type設定要畫哪種圖: 'h' = heart, 'b' = breath 
    if scatter_disp:
        plot_scatter(
            all_index_of_fftmax,
            all_gt_array, 
            all_confidenceMetricBreathOut_std,
            all_confidenceMetricBreathOut_xCorr_std,
            all_confidenceMetricBreathOut_mean,
            all_confidenceMetricBreathOut_xCorr_mean,
            all_breathingRateEst_FFT_std,all_breathingRateEst_FFT_mean,
            all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
            all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean, 
            all_sumEnergyBreathWfm_mean, 
            all_sumEnergyBreathWfm_std, 
            all_sumEnergyHeartWfm_mean, 
            all_sumEnergyHeartWfm_std)
    if knn:
        knn_test(
            predict_array,
            all_index_of_fftmax,
            all_gt_array, 
            all_confidenceMetricBreathOut_std,
            all_confidenceMetricBreathOut_xCorr_std,
            all_confidenceMetricBreathOut_mean,
            all_confidenceMetricBreathOut_xCorr_mean,
            all_breathingRateEst_FFT_std,all_breathingRateEst_FFT_mean,
            all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
            all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean, 
            all_sumEnergyBreathWfm_mean, 
            all_sumEnergyBreathWfm_std, 
            all_sumEnergyHeartWfm_mean, 
            all_sumEnergyHeartWfm_std,
            test_array1, test_array2, all_ti_og_br)