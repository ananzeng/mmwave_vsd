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
from number_analyze import heart_analyze
from losscal import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import KFold,StratifiedKFold

all_data_str = ["all_index_of_fftmax", 
            "all_confidenceMetricHeartOut_std", "all_confidenceMetricHeartOut_4Hz_std", 
            "all_confidenceMetricHeartOut_xCorr_std", "all_confidenceMetricHeartOut_mean",
            "all_confidenceMetricHeartOut_4Hz_mean", "all_confidenceMetricHeartOut_xCorr_mean",
            "all_heartRateEst_FFT_std","all_heartRateEst_FFT_mean",
            "all_heartRateEst_FFT_4Hz_std", "all_heartRateEst_FFT_4Hz_mean",
            "all_heartRateEst_xCorr_std", "all_heartRateEst_xCorr_mean",
            "all_heartRateEst_peakCount_std", "all_heartRateEst_peakCount_mean",
            "all_sumEnergyBreathWfm_mean", 
            "all_sumEnergyBreathWfm_std", 
            "all_sumEnergyHeartWfm_mean", 
            "all_sumEnergyHeartWfm_std"]
def ml_algorithm(X_train, y_train, X_test, y_test, all_data, all_gt_array):   
    kfold_test = True
    #all_data = preprocessing.scale(all_data)
    if kfold_test:
        print("-----------------------------K-Fold-----------------------------------")
        kf = StratifiedKFold(n_splits = 10, random_state = 69, shuffle = True)
        kfold_result_array = []
        for train_index, test_index in kf.split(all_data, all_gt_array):
            print("TRAIN:", train_index)
            print("TEST:", test_index)
            X_train1, X_test1 = all_data[train_index], all_data[test_index]
            y_train1, y_test1 = all_gt_array[train_index], all_gt_array[test_index]
            rf = RandomForestRegressor(n_estimators = 100, random_state = 69, n_jobs = -1, min_samples_leaf = 3, min_samples_split = 5)
            rf.fit(X_train1, y_train1)
            predictions = rf.predict(X_test1)
            round_to_whole = [round(num) for num in predictions]
            kfold = calculate_l1_loss(y_test1, round_to_whole)
            kfold_result_array.append(kfold)
            print("L1 Loss of RandomForest", kfold)
        print("avg l1 loss:", np.mean(np.array(kfold_result_array)))
    else:
        rf = RandomForestRegressor(n_estimators = 100, random_state = 69, n_jobs = -1, min_samples_leaf = 3, min_samples_split = 5)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        round_to_whole = [round(num) for num in predictions]
        importances = rf.feature_importances_
        #for index, i in enumerate(importances):
        #    print(all_data_str[index], importances[index])
        print("predict of RandomForest", round_to_whole)
        print("groundtruth", y_test)
        print("L1 Loss of RandomForest", calculate_l1_loss(y_test, round_to_whole))
        #ar = heart_analyze(round_to_whole, y_test)
    """
    xgbrmodel = xgb.XGBRegressor(n_estimators = 100, random_state = 69, n_jobs = -1)
    xgbrmodel.fit(X_train, y_train)
    predictions = xgbrmodel.predict(X_test)
    round_to_whole = [round(num) for num in predictions]

    print("predict of xgb_RandomForest", round_to_whole)
    print("groundtruth", y_test)
    print("L1 Loss of xgb_RandomForest", calculate_l1_loss(y_test, round_to_whole))
    ar = heart_analyze(round_to_whole, y_test)
    """

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


def iir_bandpass_filter_1(data, lowcut, highcut, signal_freq, filter_order, ftype):
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

def detect_breath(unw_phase):
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
    std_of_phase_diff = np.std(np.array(phase_diff))
    #print("STD of phase_diff", std_of_phase_diff)
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

    # butter ellip cheby2
    bandpass_sig = iir_bandpass_filter_1(new_phase_diff, 0.125, 0.55, 20, 5, "cheby2") # 0.8 2.1 
    N = len(bandpass_sig)
    T = 1 / 20
    bps_fft = fft(bandpass_sig)
    bps_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)    
    #print(np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2))
    index_of_fftmax = np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2)
    #print("index_of_fftmax", index_of_fftmax)
    if index_of_fftmax < 0.215:
        replace = True
    # Smoothing signal
    smoothing_signal = MLR(bandpass_sig, 2)  # Breath = 9, Heart = 6, Delta = 1

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
    #print(f'Rate: {rate}')

    return rate, replace, index_of_fftmax, std_of_phase_diff

if __name__ == '__main__':

    ml = True
    all_gt_array = []

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


    predict_array = []
    numofsample = 0
    type = ["train", "test"]
    for data_type in type:
        for user in os.listdir(os.path.join("rf", data_type)):
            if os.path.isdir(os.path.join("rf", data_type, user, "gt_br")):
                files_path = os.path.join("rf", data_type, user, "0.8")
                #print(files_path)
                ground_truth_files_path = os.path.join("rf", data_type, user, "gt_br")
                #print(ground_truth_files_path)
                files = os.listdir(files_path)        

                for  name  in os.listdir(ground_truth_files_path):
                    with open(os.path.join(ground_truth_files_path, name)) as f:
                        for line in f.readlines():
                            all_gt_array.append(int(line))
                if data_type == "train":
                    numofsample = len(all_gt_array)
                    #print("numofsample：", numofsample)
                for tmp in range(0, len(files)//2, 1):
                    file = files[tmp]
                    #print(f'\nCurrent file: {file}')
                    datas_path = os.path.join(files_path, file)
                    vitial_sig = pd.read_csv(datas_path)
                    unwrapPhase = vitial_sig['unwrapPhasePeak_mm'].values
                    heart = vitial_sig['rsv[0]'].values
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

                    for i in range (0, 800, 800):  # 0, 600, 1200
                        result_rate, replace1, index_of_fftmax, std_of_phase_diff = detect_breath(unwrapPhase[0 + i: 800 + i])
                        #if replace1:
                        #    result_rate = int(np.mean(heart))  

                        all_std_of_phase_diff.append(std_of_phase_diff)
                        all_index_of_fftmax.append(index_of_fftmax)
                        predict_array.append(round(result_rate))

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

    all_data = [all_index_of_fftmax, 
                all_confidenceMetricBreathOut_std, all_confidenceMetricBreathOut_xCorr_std,
                all_confidenceMetricBreathOut_mean, all_confidenceMetricBreathOut_xCorr_mean,
                all_breathingRateEst_FFT_std, all_breathingRateEst_FFT_mean,
                all_breathingEst_xCorr_std, all_breathingEst_xCorr_mean,
                all_breathingEst_peakCount_std, all_breathingEst_peakCount_mean,
                all_sumEnergyBreathWfm_mean, 
                all_sumEnergyBreathWfm_std, 
                all_sumEnergyHeartWfm_mean, 
                all_sumEnergyHeartWfm_std]

    all_data = np.array(all_data).transpose()
    all_gt_array = np.array(all_gt_array)
    
    X_train = all_data[:numofsample][:]
    y_train = all_gt_array[:numofsample]

    X_test = all_data[numofsample:]
    y_test = all_gt_array[numofsample:]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
        # ML
    if ml:
        ml_algorithm(X_train, y_train, X_test, y_test, all_data, all_gt_array)