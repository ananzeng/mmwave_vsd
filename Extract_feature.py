import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import combine_svm
import scipy.signal as signal
import scipy
sns.set()

def feature_detection(smoothing_signal):
    feature_peak, _ = signal.find_peaks(smoothing_signal)
    feature_valley, _ = signal.find_peaks(-smoothing_signal)
    data_value = np.multiply(np.square(smoothing_signal), np.sign(smoothing_signal))
    return feature_peak, feature_valley, data_value

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

# fRSA (30sec)
def fRSA_fn(br_sig):
    auto_br_sig = np.correlate(br_sig, br_sig, mode='full')
    auto_br_sig = auto_br_sig[auto_br_sig.size//2:]
    feature_peak, feature_valley, _= feature_detection(auto_br_sig)
    fRSA = caculate_breathrate(feature_peak, feature_valley)
    fRSA = (fRSA / 2)
    return fRSA

# 10個fRSA
def tfRSA_fn(fRSA_sig):
    tfRSA = np.std(fRSA_sig)
    return tfRSA

# 31個f𝑅𝑆𝐴
def sfRSA_fn(fRSA_sig):
    sfRSA = scipy.signal.savgol_filter(fRSA_sig, 31, 3)
    sfRSA_mean = np.average(sfRSA)
    return sfRSA, sfRSA_mean

# 31個tf𝑅𝑆𝐴
def stfRSA_fn(tfRSA_sig):
    stfRSA = scipy.signal.savgol_filter(tfRSA_sig, 31, 3)
    stfRSA_mean = np.average(stfRSA)
    return stfRSA, stfRSA_mean

# 31個f𝑅𝑆𝐴
def sdfRSA_fn(fRSA, sfRSA):
    sdfRSA = np.abs(fRSA - sfRSA)
    sdfRSA = scipy.signal.savgol_filter(sdfRSA, 31, 3)
    return sdfRSA

if __name__=="__main__":

    # Data path
    names = "./dataset_sleep"
    for name in os.listdir(names):
        files = os.path.join(names, name, "0.8")
        for num in range(len(os.listdir(files)) // 2):
            datas = os.listdir(files)[num]
            data = os.path.join(files, datas)
            raw_data = pd.read_csv(data)
            
            # Data preprocessing
            unwrap_phase = raw_data["unwrapPhasePeak_mm"]
            phase_diff = combine_svm.Phase_difference(unwrap_phase)
            re_phase_diff = combine_svm.Remove_impulse_noise(phase_diff, 1.5)
            amp_sig = combine_svm.Amplify_signal(re_phase_diff)  # Consider deleting
            bandpass_sig = combine_svm.iir_bandpass_filter_1(amp_sig, 0.125, 0.55, 20, 5, "cheby2")

            # Sliding window
            loacl_fRSA = []
            for index in range(len(bandpass_sig) - 30*20):
                window = bandpass_sig[index:index + 30*20]
                fRSA = fRSA_fn(window)
                loacl_fRSA.append(fRSA)

            
                
                

