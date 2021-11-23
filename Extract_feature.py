import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import combine_svm
import pandas as pd
from tqdm import tqdm
import scipy.signal as signal
import csv
import seaborn as sns
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

''' ----------------------- Movement ----------------------- '''
# 30s訊號 = 30 * 20 = 600
def mov_dens_fn(raw_sig):
    new_sig = []
    count = 0
    for num in range(80):
        top = 0
        first = int(num*(0.5*20))
        last = int((num+1)*(0.5*20))
        x = raw_sig[first:last]

        # 方差公式
        for i in range(10):
            top += np.square(x[i] - np.average(x))
        result = top / (10 - 1)
        if result > 0.005:  # 閥值可調
            count += 1
    percent = (count/80) * 100
    return percent

''' ----------------------- Respiration ----------------------- '''
# fRSA (40sec)
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
    stfRSA = scipy.signal.savgol_filter(np.array(tfRSA_sig), 31, 2)
    stfRSA_mean = np.average(stfRSA)
    return stfRSA, stfRSA_mean

# 31個f𝑅𝑆𝐴
def sdfRSA_fn(fRSA, sfRSA):
    sdfRSA = np.abs(fRSA - sfRSA)
    sdfRSA = scipy.signal.savgol_filter(sdfRSA, 31, 3)
    sdfRSA_mean = np.average(sdfRSA)
    return sdfRSA, sdfRSA_mean

''' ----------------------- Heart rate ----------------------- '''
def mHR_fn(heart):
    return fRSA_fn(heart)

# 待補
def LF_HF_LFHF(sig):
    LF_sig = combine_svm.iir_bandpass_filter_1(amp_sig, 0.04, 0.15, 20, 9, "cheby2")
    HF_sig = combine_svm.iir_bandpass_filter_1(amp_sig, 0.15, 0.4, 20, 9, "cheby2")

def tmHR_fn(mHR_sig):
    return tfRSA_fn(mHR_sig)

def smHR_fn(mHR_sig):
    return sfRSA_fn(mHR_sig)

def stmHR_fn(tmHR_sig):
    return stfRSA_fn(tmHR_sig)

def sdmHR_fn(mHR, smHR):
    return sdfRSA_fn(mHR, smHR)


# 待補
def sHF_fn(HF_sig):
    sHF = scipy.signal.savgol_filter(HF_sig, 31, 3)
    sHF_mean = np.average(sfRSA)
    return sHF, sHF_mean

def sLFHF_fn(LFHF_sig):
    sLFHF = scipy.signal.savgol_filter(LFHF_sig, 31, 3)
    sLFHF_mean = np.average(sfRSA)
    return sLFHF, sLFHF_mean
    
if __name__=="__main__":

    # Data path
    names = "./dataset_sleep"
    for name in os.listdir(names):
        files = os.path.join(names, name, "0.8")
        for num in range(len(os.listdir(files)) // 2):
            datas = os.listdir(files)[num]
            data = os.path.join(files, datas)
            raw_data = pd.read_csv(data)
            raw_data_pd = pd.DataFrame(raw_data)
            
            # Data preprocessing
            unwrap_phase = raw_data["unwrapPhasePeak_mm"]
            # breath_energy = raw_data["sumEnergyBreathWfm"]
            heart_energy = raw_data["sumEnergyHeartWfm"]
            phase_diff = combine_svm.Phase_difference(unwrap_phase)
            re_phase_diff = combine_svm.Remove_impulse_noise(phase_diff, 1.5)
            amp_sig = combine_svm.Amplify_signal(re_phase_diff)  # Consider deleting
            breath_sig = combine_svm.iir_bandpass_filter_1(amp_sig, 0.125, 0.55, 20, 5, "cheby2")  # [:1000]
            heart_sig = combine_svm.iir_bandpass_filter_1(amp_sig, 0.9, 1.9, 20, 9, "cheby2")  # [:1000]

            # Array
            loacl_fRSA = []
            loacl_tfRSA = []
            loacl_mHR = []
            loacl_tmHR = []

            # Respiration
            local_stfRSA_mean = []
            local_sfRSA_mean = []
            local_sdfRSA_mean = []

            # Heart rate
            local_stmHR_mean = []
            local_smHR_mean = []
            local_sdmHR_mean = []

            # Sliding window
            for index in tqdm(range(len(breath_sig) - 40*20 + 1)):
                window_b = breath_sig[index:index + 40*20]
                window_h = heart_sig[index:index + 40*20]

                # ---------- Movement ---------- 
                mov_dens = mov_dens_fn(window_b)

                # ---------- Respiration ---------- 
                fRSA = fRSA_fn(window_b)
                loacl_fRSA.append(fRSA)
                if len(loacl_fRSA) >= 10:
                    tfRSA = tfRSA_fn(loacl_fRSA[-10:])
                    loacl_tfRSA.append(tfRSA)
                    if len(loacl_tfRSA) >= 31:
                        stfRSA, stfRSA_mean = stfRSA_fn(loacl_tfRSA[-31:])
                        local_stfRSA_mean.append(stfRSA_mean)
                if len(loacl_fRSA) >= 31:
                        sfRSA, sfRSA_mean = sfRSA_fn(loacl_fRSA[-31:])
                        local_sfRSA_mean.append(sfRSA_mean)
                        sdfRSA, sdfRSA_mean = sdfRSA_fn(loacl_fRSA[-31:], sfRSA[-31:])
                        local_sdfRSA_mean.append(sdfRSA_mean)
                
                # ---------- Heart rate ---------- 
                mHR = mHR_fn(window_h)
                loacl_mHR.append(mHR)
                if len(loacl_mHR) >= 10:
                    tmHR = tmHR_fn(loacl_mHR[-10:])
                    loacl_tmHR.append(tmHR)
                    if len(loacl_tmHR) >= 31:
                        stmHR, stmHR_mean = stmHR_fn(loacl_tmHR[-31:])
                        local_stmHR_mean.append(stmHR_mean)
                if len(loacl_mHR) >= 31:
                        smHR, smHR_mean = smHR_fn(loacl_mHR[-31:])
                        local_smHR_mean.append(smHR_mean)
                        sdmHR, sdmHR_mean = sdmHR_fn(loacl_mHR[-31:], smHR[-31:])
                        local_sdmHR_mean.append(sdmHR_mean)

            # 補植 (維持長度)
            layer0 = 40*20-1
            layer1 = layer0 + 10 - 1
            layer2 = layer1 + 31 - 1
            layer3 = layer0 + 31 - 1
            for i in range(layer0):
                loacl_fRSA.insert(i, 0)
                loacl_mHR.insert(i, 0)
            for j in range(layer1):
                loacl_tfRSA.insert(j, 0)
                loacl_tmHR.insert(j, 0)
            for k in range(layer2):
                local_stfRSA_mean.insert(k, 0)
                local_stmHR_mean.insert(k, 0)
            for l in range(layer3):
                local_sfRSA_mean.insert(l, 0)
                local_sdfRSA_mean.insert(l, 0)
                local_smHR_mean.insert(l, 0)
                local_sdmHR_mean.insert(l, 0)

            # 寫入CSV
            # raw_data_pd = raw_data_pd[:1000]

            # Layer0
            raw_data_pd.insert(38, "fRSA", loacl_fRSA)
            raw_data_pd.insert(39, "mHR", loacl_mHR)

            # Layer1
            raw_data_pd.insert(40, "tfRSA", loacl_tfRSA)
            raw_data_pd.insert(41, "tmHR", loacl_tmHR)

            # Layer2
            raw_data_pd.insert(42, "stfRSA", local_stfRSA_mean)
            raw_data_pd.insert(43, "stmHR", local_stmHR_mean)
    
            # Layer3
            raw_data_pd.insert(44, "sfRSA", local_sfRSA_mean)
            raw_data_pd.insert(45, "sdfRSA", local_sdfRSA_mean)
            raw_data_pd.insert(46, "smHR", local_smHR_mean)
            raw_data_pd.insert(47, "sdmHR", local_sdmHR_mean)

            test = raw_data_pd
            test.to_csv("./sleep_features/test.csv", index=False)
            print(test)
            
                
                
