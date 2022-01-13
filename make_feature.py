import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import scipy
import combine_svm
import datetime
import pandas as pd
from tqdm import tqdm
import csv
import scipy.signal
from scipy.fftpack import fft
import seaborn as sns
from combine_svm import Phase_difference, Remove_impulse_noise, Amplify_signal, iir_bandpass_filter_1, MLR, feature_detection, feature_compress, candidate_search

""" --------------------------------------------- KNN Features --------------------------------------------- """
# --------------------------------------- Body Movement Index (BMI) ---------------------------------------
def bmi(sig):
    """
    Return bmi array of given raw_sig.

    Parameters
    ----------
    start_min & start_sec:
        紀錄測量到的第一個數據
    coco:
        判斷是否已經完成忽略起始未滿 60 秒的數據
    number_i:
        計算 Sampling rate 的 Index
    bi_arr:
        最後輸出的 BMI 陣列
    amp_sec:
        計算每秒輸出的平均，因為取樣頻率有差異所以這樣做
    window_sig:
        存放長度為 60 秒的資料
    window_sig_piece:
        以大小為 10s 的窗格處理 window_sig 因此長度為 6

    Returns
    -------
    out : ndarray
        bmi array of given raw_sig.
    """
    start_min = sig.datetime[0][3:5]
    start_sec = sig.datetime[0][6:8]
    coco = False
    number_i = 0
    bi_arr = []
    window_sig = []
    window_sig_piece = []
    sampling_rate_arr = np.ones(60)
    for index in tqdm(range(len(sig))):

        # 為了保持 60 秒的時間區間，一開始當秒數不等於 00 時皆刪去
        if coco == False and start_sec != "00":
            start_min = sig.datetime[index][3:5]
            start_sec = sig.datetime[index][6:8]
            bi_arr.append(0)
        else:
            coco = True

        if coco:
            if start_min == sig.datetime[index][3:5]:
                window_sig.append(sig.unwrapPhasePeak_mm[index])
                if start_sec == sig.datetime[index][6:8]:
                    sampling_rate_arr[number_i] += 1
                    bi_arr.append(0)
                else:
                    number_i += 1
                    start_sec = sig.datetime[index][6:8]
                    bi_arr.append(0)

            else:
                start_min = sig.datetime[index][3:5]
                start_sec = sig.datetime[index][6:8]
                amp_sec = []
                num_start = 0
                for numbers in sampling_rate_arr:
                    num_end = int(numbers) + num_start
                    amp_sec.append(np.mean(window_sig[num_start:num_end]))  # Paper 沒寫暫時用 mean
                    num_start = num_end

                window_size = 10
                for i in range(6):
                    ak = np.mean(amp_sec[i*window_size:(i+1)*window_size])  # Paper 沒寫暫時用 mean
                    window_sig_piece.append(ak)

                    if i == 0:
                        a_min = ak
                    if a_min > ak:
                        a_min = ak

                # 計算BI(k)
                bi = 0
                for i in range(6):
                    tmp = window_sig_piece[i]
                    bi += (tmp - a_min)

                # 得到的特徵
                bi_arr.append(bi)

                # Reset
                number_i = 0
                window_sig = []
                window_sig_piece = []
                sampling_rate_arr = np.ones(60)
    return bi_arr

# --------------------------------------- Variance of RPM --------------------------------------- 

# 需先將單位改成秒
def var_RPM(sig, mod):
    if mod == 0:
        breath = sig["breath"]
    else:
        breath = sig["heart"]
    N = 10 * 60  # 預設為 10min = 10 * 60s
    var_RPM_k = []

    # 過程補 0
    for i in range(N):
        var_RPM_k.append(0)

    # 窗格大小: 10 (窗格重疊)
    for k in tqdm(range(len(breath) - N)):
        window_sum = 0
        tmp = np.copy(breath[k:k+N])
        tmp_mean = np.mean(tmp)
        for j in range(N):
            tmp_a = tmp[j] - tmp_mean
            if tmp_a == 0:
                window_sum += 0
            else:
                window_sum += np.square(tmp_a)
        var_RPM_k.append(window_sum / (N - 1))

    # 窗格大小: 10 (窗格不重疊)
    # for k in range(len(sig) // N):
    #     window = 0
    #     tmp = breath[k*10:(k+1)*10]
    #     tmp_mean = np.mean(tmp)
    #     for j in range(N):
    #         window += np.square(tmp[j] - tmp_mean)
    #     var_RPM_k.append(window / (N - 1))
    
    return var_RPM_k

# --------------------------------------- Amplitude Difference Accumulation (ADA) of Respiration --------------------------------------- 
def ada_assist(sig, brhr):

    # 0: 呼吸, 1: 心跳
    if brhr == 0:
        a = [1.5, 0.125, 0.55, 20, 5, 2, 22, 17]
    else:
        a = [1.5, 0.9, 1.9, 20, 9, 2, 5, 4]

    # Phase difference
    phase_diff = Phase_difference(sig)

    # RemoveImpulseNoise
    re_phase_diff = Remove_impulse_noise(phase_diff, int(a[0]))

    # Linear amplify
    amp_sig = Amplify_signal(re_phase_diff)

    # Bandpass signal (cheby2)
    bandpass_sig = iir_bandpass_filter_1(amp_sig, float(a[1]), float(a[2]), int(a[3]), int(a[4]), "cheby2") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
 
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
    NT_points, _ = candidate_search(smoothing_signal, compress_feature, int(a[7]))  # breath = 18 hreat = 4 ex: 7

    return NT_points

def ada(sig, brhr):
    """
    每分鐘更新，ADA
    """
    unwrapPhase = sig['unwrapPhasePeak_mm'].values
    start_min = sig.datetime[0][3:5]
    start_sec = sig.datetime[0][6:8]
    coco = False
    ada_arr = []
    window_sig = []
    number_i = 0
    sampling_rate_arr = np.ones(60)
    for index in tqdm(range(len(unwrapPhase))):

        # 為了保持 60 秒的時間區間，一開始當秒數不等於 00 時皆刪去
        if coco == False and start_sec != "00":
            start_min = sig.datetime[index][3:5]
            start_sec = sig.datetime[index][6:8]
            ada_arr.append(0)
        else:
            coco = True

        # ----------------------------------------------------
        if coco:
            if start_min == sig.datetime[index][3:5]:
                window_sig.append(sig.unwrapPhasePeak_mm[index])

                if start_sec == sig.datetime[index][6:8]:
                    sampling_rate_arr[number_i] += 1
                    ada_arr.append(0)
                    
                else:
                    number_i += 1
                    start_sec = sig.datetime[index][6:8]
                    ada_arr.append(0)
            else:
                diff = 0
                start_min = sig.datetime[index][3:5]
                start_sec = sig.datetime[index][6:8]
                top = ada_assist(window_sig, brhr)  # 0 for breath, 1 for heart.
                for top_index in range(len(top) - 1):
                    cur_index = int(top[top_index + 1])
                    pre_index = int(top[top_index])
                    diff += math.fabs(window_sig[cur_index] - window_sig[pre_index])
                ada_arr.append(diff)
                number_i = 0
                window_sig = []
                sampling_rate_arr = np.ones(60)
    return ada_arr

# --------------------------------------- REM Parameter --------------------------------------- 
# 需先將單位改成秒
def rem_parameter(sig):
    q = 2
    start_sec = sig.datetime[0][6:8]
    coco = False
    resp = sig["breath"].values

    count = 0
    rem_arr = []
    forward_win = []
    backward_win = []
    tmp_arr = []  # 用來放平滑化之前的

    for index in tqdm(range(len(resp))):

        # 為了保持 60 秒的時間區間，一開始當秒數不等於 00 時皆刪去
        if coco == False and start_sec != "00":
            start_sec = sig.datetime[index][6:8]
            rem_arr.append(0)
        else:
            coco = True

        # ----------------------------------------------------
        if coco:
            if count <= 29:
                forward_win.append(resp[index])
                count += 1
                if count == 29:
                    backward_win.append(resp[index])
            elif 29 <= count and count < 59:
                backward_win.append(resp[index])
                count += 1
            else:
                forward_resp = np.mean(forward_win)
                backward_resp = np.mean(backward_win)
                tmp_arr.append(np.abs(forward_resp - backward_resp))
                count = 0

                forward_win = []
                backward_win = []
                
    
    for tmp in range(q):
        for i in range(60):
            rem_arr.append(tmp_arr[tmp])
        
    for index_rem in tqdm(range(q, len(tmp_arr) - q)):
        for i in range(60):
            rem_arr.append(np.mean(tmp_arr[index_rem - q:index_rem + q + 1]))

    for tmp in range(q):
        for i in range(60):
            rem_arr.append(tmp_arr[len(tmp_arr) - q + tmp])

    # 補足長度 (不足 1分鐘補 0)
    res = len(resp) - len(rem_arr)
    for tmp in range(res):
        rem_arr.append(0)

    return rem_arr

# 需放 BMI 後
def deep_parameter(sig):
    pre = 0
    deep_p = []
    tmp_sig = sig
    bmi_index = pd.DataFrame(tmp_sig["bmi"].values)
    bmi_index = bmi_index.replace(0, np.nan).dropna()
    # print("Orignal: ", len(sig["bmi"].values))
    bmi = pd.DataFrame(sig["bmi"].values)
    heart = pd.DataFrame(sig["heart"].values)
    deep_p.append(0)
    for tmp in tqdm(bmi_index.index):
        pad = tmp - pre - 1
        for i in range(pad):
            deep_p.append(0)
        deep_p.append(bmi.loc[tmp].values[0] / (bmi.loc[tmp].values[0] + heart.loc[tmp].values[0]))
        pre = tmp

    # 補足長度 (不足 1分鐘補 0)
    res = len(bmi) - len(deep_p)
    for tmp in range(res):
        deep_p.append(0)
    # print("After: ", len(deep_p))
    return deep_p

""" --------------------------------------------- KNN Features --------------------------------------------- """


def mov_dens_fn(raw_sig):
    count = 0
    # Segments with 0.5s length = 80 Segments
    for num in range(10):  # 80
        top = []
        first = int(num*(4))  # 0.5*20
        last = int((num+1)*(4))  # 0.5*20
        x = np.array(np.round(raw_sig[first:last], 8))
        # 方差公式
        for i in range(4):
            top.append(np.square(x[i] - np.average(x)))
        result = np.sum(top) / (4 - 1)
        if result > 0.045:  # 閥值可調
            count += 1
    percent = (count/10) * 100  # 80
    return percent

''' ----------------------- Respiration ----------------------- '''
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
def tmHR_fn(mHR_sig):
    return tfRSA_fn(mHR_sig)

def smHR_fn(mHR_sig):
    return sfRSA_fn(mHR_sig)

def stmHR_fn(tmHR_sig):
    return stfRSA_fn(tmHR_sig)

def sdmHR_fn(mHR, smHR):
    return sdfRSA_fn(mHR, smHR)

''' ----------------------- HRV ----------------------- '''
def LF_HF_LFHF(sig):
    LF_sig = combine_svm.iir_bandpass_filter_1(sig, 0.04, 0.15, 20, 2, "cheby2")
    HF_sig = combine_svm.iir_bandpass_filter_1(sig, 0.15, 0.4, 20, 2, "cheby2")
    LF_eng = energe(LF_sig)
    HF_eng = energe(HF_sig)
    LFHF_eng = HF_eng / LF_eng
    return LF_eng, HF_eng, LFHF_eng

def energe(sig):
    N = len(sig)
    bps_fft = np.fft.fft(sig)
    return np.sum(np.square(np.abs(bps_fft[:N // 2])))

def sHF_fn(HF_sig):
    sHF = scipy.signal.savgol_filter(HF_sig, 31, 3)
    sHF_mean = np.average(sHF)
    return sHF, sHF_mean

def sLFHF_fn(LFHF_sig):
    sLFHF = scipy.signal.savgol_filter(LFHF_sig, 31, 3)
    sLFHF_mean = np.average(sLFHF)
    return sLFHF, sLFHF_mean

if __name__ == "__main__":
    raw_data_pd = pd.DataFrame()
    names = "./dataset_sleep_test"
    files = os.path.join(names, "processed_data")
    #files = "processed_data"
    print(files)
    for num in range(len(os.listdir(files))):
        datas = os.listdir(files)[num]
        data = os.path.join(files, datas)
        print(data)
        raw_data_pd = pd.read_csv(data)

        # ------------- KNN Features ------------- 
        # Body Movement Index (BMI)
        bmi = bmi(raw_data_pd)
        raw_data_pd["bmi"] = bmi

        # Deep Parameter
        deep_p = deep_parameter(raw_data_pd)
        raw_data_pd["deep_p"] = deep_p

        # Amplitude Difference Accumulation (ADA) of Respiration
        ada_arr = ada(raw_data_pd, 0)  # brhr: 0(呼吸) 1(心跳)
        raw_data_pd["ada_br"] = ada_arr

        # Amplitude Difference Accumulation (ADA) of Heartbeat
        ada_arr = ada(raw_data_pd, 1)  # brhr: 0(呼吸) 1(心跳)
        raw_data_pd["ada_hr"] = ada_arr

        # ------------- KNN Features -------------  
        raw_data_pd['heart'] = raw_data_pd['heart'].replace(0, np.nan)
        new_data = raw_data_pd.dropna().reset_index()

        #  Variance of RPM
        var_RPM_k = var_RPM(new_data, 0)  # mod: 0(呼吸) 1(心跳)
        new_data["var_RPM"] = var_RPM_k

        #  Variance of HPM
        var_RPM_k = var_RPM(new_data, 1)  # mod: 0(呼吸) 1(心跳)
        new_data["var_HPM"] = var_RPM_k

        # REM Parameter
        rem = rem_parameter(new_data)
        new_data["rem_parameter"] = rem     

        # ------------------------ Drop ------------------------
        breath = np.array(new_data['breath'])
        heart = np.array(new_data['heart'])
        outputFilterBreathOut = np.array(new_data['outputFilterBreathOut'])
        outputFilterHeartOut = np.array(new_data['outputFilterHeartOut'])
        heartRateEst_FFT = np.array(new_data['heartRateEst_FFT'])
        heartRateEst_FFT_4Hz = np.array(new_data['heartRateEst_FFT_4Hz'])
        heartRateEst_xCorr = np.array(new_data['heartRateEst_xCorr'])
        heartRateEst_peakCount = np.array(new_data['heartRateEst_peakCount'])
        breathingRateEst_FFT = np.array(new_data['breathingRateEst_FFT'])
        breathingEst_xCorr = np.array(new_data['breathingEst_xCorr'])
        breathingEst_peakCount = np.array(new_data['breathingEst_peakCount'])
        confidenceMetricBreathOut = np.array(new_data['confidenceMetricBreathOut'])
        confidenceMetricBreathOut_xCorr = np.array(new_data['confidenceMetricBreathOut_xCorr'])
        confidenceMetricHeartOut = np.array(new_data['confidenceMetricHeartOut'])
        confidenceMetricHeartOut_4Hz = np.array(new_data['confidenceMetricHeartOut_4Hz'])
        confidenceMetricHeartOut_xCorr = np.array(new_data['confidenceMetricHeartOut_xCorr'])
        sumEnergyBreathWfm = np.array(new_data['sumEnergyBreathWfm'])
        sumEnergyHeartWfm = np.array(new_data['sumEnergyHeartWfm'])
        rsv_0 = np.array(new_data['rsv[0]'])
        rsv_1 = np.array(new_data['rsv[1]'])
        print(f"Total len: {len(heart)}\n")

        # ------------------------ Mov_dens ------------------------ 
        unwrapPhasePeak_mm = new_data["unwrapPhasePeak_mm"]

        local_mov_dens = []
        num_in_window = 40  # 40s

        # 空值補0
        for i in range(num_in_window-1):
            local_mov_dens.append(0)

        for turn in tqdm(range(len(unwrapPhasePeak_mm) - num_in_window + 1)):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            window_mov_dens = unwrapPhasePeak_mm[start_index:end_index]

            # Standard deviation
            mov_dens = mov_dens_fn(window_mov_dens)
            local_mov_dens.append(str(round(mov_dens, 4)))

        print(f"Real len: {len(unwrapPhasePeak_mm)}\nMov_dens len: {len(local_mov_dens)}")

        new_data["mov_dens"] = local_mov_dens

        # ------------- HRV ------------- 
        local_LF = []
        local_HF = []
        local_LFHF = []
        num_in_window = 60*5  # 5 min

        # 空值補0
        for i in range(num_in_window-1):
            local_LF.append(0)
            local_HF.append(0)
            local_LFHF.append(0)

        for turn in range(len(unwrapPhasePeak_mm) - num_in_window + 1):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            window_unwrapPhasePeak_mm = unwrapPhasePeak_mm[start_index:end_index]

            # LF_HF_LFHF
            LF, HF, LFHF= LF_HF_LFHF(window_unwrapPhasePeak_mm)
            local_LF.append(str(round(LF, 4)))
            local_HF.append(str(round(HF, 4)))
            local_LFHF.append(str(round(LFHF, 4)))

        print(f"Real len: {len(unwrapPhasePeak_mm)}\nLF len: {len(local_LF)}\nHF len: {len(local_HF)}\nLFHF len: {len(local_LFHF)}")
        new_data["LF"] = local_LF
        new_data["HF"] = local_HF
        new_data["LFHF"] = local_LFHF

        # ------------------------ sHF sLFHF ------------------------ 
        HF = new_data["HF"]
        LFHF = new_data["LFHF"]
        local_sHF = []
        local_sLFHF = []
        num_in_window = 31

        # 空值補0
        for i in range(num_in_window-1):
            local_sHF.append(0)
            local_sLFHF.append(0)

        for turn in range(len(HF) - num_in_window + 1):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            _, window_HF = sHF_fn(HF[start_index:end_index])
            _, window_LFHF = sLFHF_fn(LFHF[start_index:end_index])

            # SG filter
            local_sHF.append(str(round(window_HF, 4)))
            local_sLFHF.append(str(round(window_LFHF, 4)))
        print(f"Real len: {len(HF)}\nLF len: {len(local_LF)}\nsHF len: {len(local_sHF)}\nsLFHF len: {len(local_sLFHF)}")
        new_data["sHF"] = local_sHF
        new_data["sLFHF"] = local_sLFHF

        # ------------- tfRSA and tmHR ------------- 
        local_tfRSA = []
        local_tmHR = []
        num_in_window = 10

        # 空值補0
        for i in range(num_in_window-1):
            local_tfRSA.append(0)
            local_tmHR.append(0)

        for turn in range(len(breath) - num_in_window + 1):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            window_breath = breath[start_index:end_index]
            window_heart = heart[start_index:end_index]

            # Standard deviation
            local_tfRSA.append(str(round(tfRSA_fn(window_breath), 4)))
            local_tmHR.append(str(round(tmHR_fn(window_heart), 4)))

        print("tfRSA and tmHR Complete!")
        print(f"Real len: {len(breath)}\ntfRSA len: {len(local_tfRSA)}\ntmHR len: {len(local_tmHR)}")

        #  ------------- sfRSA and smHR and sdfRSA and sdmHR------------- 
        local_sfRSA = []
        local_sdfRSA = []
        local_smHR = []
        local_sdmHR = []
        num_in_window = 31

        # 空值補0
        for i in range(num_in_window-1):
            local_sfRSA.append(0)
            local_smHR.append(0)
            local_sdfRSA.append(0)
            local_sdmHR.append(0)

        for turn in range(len(breath) - num_in_window + 1):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            window_breath = breath[start_index:end_index]
            window_heart = heart[start_index:end_index]

            # Savitzky–Golay filter
            sfRSA, sfRSA_mean = sfRSA_fn(window_breath)
            sdfRSA, sdfRSA_mean = sdfRSA_fn(window_breath, sfRSA)
            smHR, smHR_mean = smHR_fn(window_heart)
            sdmHR, sdmHR_mean = sdmHR_fn(window_heart, smHR)
            local_sfRSA.append(round(sfRSA_mean, 4))
            local_sdfRSA.append(round(sdfRSA_mean, 4))
            local_smHR.append(round(smHR_mean, 4))
            local_sdmHR.append(round(sdmHR_mean, 4))

        print("\nsfRSA and smHR Complete!")
        print(f"Real len: {len(heart)}\nsfRSA len: {len(local_sfRSA)}\nsmHR len: {len(local_smHR)}")

        # 插入特徵
        new_data["tfRSA"] = local_tfRSA
        new_data["tmHR"] = local_tmHR
        new_data["sfRSA"] = local_sfRSA
        new_data["smHR"] = local_smHR
        new_data["sdfRSA"] = local_sdfRSA
        new_data["sdmHR"] = local_sdmHR


        #new_data['tmHR'] = new_data['tmHR'].replace(0, np.nan)
        #new_data = new_data.dropna()

        tfRSA = np.array(new_data['tfRSA'])
        tmHR = np.array(new_data['tmHR'])

        #  ------------- stfRSA and stmHR ------------- 
        local_stfRSA = []
        local_stmHR = []
        num_in_window = 31

        # 空值補0
        for i in range(num_in_window-1):
            local_stfRSA.append(0)
            local_stmHR.append(0)

        for turn in range(len(tfRSA) - num_in_window + 1):
            start_index = turn
            end_index = start_index + num_in_window

            # Slide data
            window_tfRSA = tfRSA[start_index:end_index]
            window_tmHR = tmHR[start_index:end_index]

            # Savitzky–Golay filter
            stfRSA, stfRSA_mean = stfRSA_fn(window_tfRSA)
            stmHR, stmHR_mean = stmHR_fn(window_tmHR)
            local_stfRSA.append(round(stfRSA_mean, 4))
            local_stmHR.append(round(stmHR_mean, 4))

        new_data["stfRSA"] = local_stfRSA
        new_data["stmHR"] = local_stmHR

        print("\nsfRSA and smHR Complete!")
        print(f"Real len: {len(heart)}\nstfRSA len: {len(local_stfRSA)}\nstmHR len: {len(local_stmHR)}")

        #  ------------- all_time ------------- 
        all_time = []
        datetime_array = np.array(new_data['datetime'])
        time_800 = datetime.datetime.strptime('20:00:00', "%H:%M:%S")
        for i in datetime_array:
            #print(i)
            i = datetime.datetime.strptime(str(i), "%H:%M:%S")
            time_feature = i - time_800
            #print(time_feature.seconds)
            all_time.append(time_feature.seconds)
        new_data["time"] = all_time

        print("\ntime Complete!")
        print(f"Real len: {len(heart)}\ntime len: {len(all_time)}")

        #  ------------- sleep_counter -------------
        stage_counter = []
        stage = np.array(new_data['sleep'])
        stage_5 = 0
        stage_4 = 0
        stage_3 = 0
        stage_2 = 0   
        temp_stage = 0     
        for i in stage:
            #print(i)
            if i == 5:
                if temp_stage != i:
                    stage_5 = 0
                stage_5 += 1
                stage_counter.append(stage_5)
                temp_stage = int(i)
            elif i == 4:
                if temp_stage != i:
                    stage_4 = 0                
                stage_4 += 1
                stage_counter.append(stage_4)
                temp_stage = int(i)
            elif i == 3:
                if temp_stage != i:
                    stage_3 = 0                
                stage_3 += 1
                stage_counter.append(stage_3)
                temp_stage = int(i)
            elif i == 2:
                if temp_stage != i:
                    stage_2 = 0                
                stage_2 += 1
                stage_counter.append(stage_2)
                temp_stage = int(i)
        print(len(stage_counter))
        new_data["sleep_counter"] = stage_counter
        print("\ntime Complete!")
        print(f"Real len: {len(heart)}\ntime len: {len(all_time)}")

        """
        new_data["outputFilterBreathOut"] = outputFilterBreathOut
        new_data["outputFilterHeartOut"] = outputFilterHeartOut
        new_data["heartRateEst_FFT"] = heartRateEst_FFT
        new_data["heartRateEst_FFT_4Hz"] = heartRateEst_FFT_4Hz
        new_data["heartRateEst_xCorr"] = heartRateEst_xCorr
        new_data["heartRateEst_peakCount"] = heartRateEst_peakCount
        new_data["breathingRateEst_FFT"] = breathingRateEst_FFT
        new_data["breathingEst_xCorr"] = breathingEst_xCorr
        new_data["breathingEst_peakCount"] = breathingEst_peakCount
        new_data["confidenceMetricBreathOut"] = confidenceMetricBreathOut
        new_data["confidenceMetricBreathOut_xCorr"] = confidenceMetricBreathOut_xCorr
        new_data["confidenceMetricHeartOut"] = confidenceMetricHeartOut
        new_data["confidenceMetricHeartOut_4Hz"] = confidenceMetricHeartOut_4Hz
        new_data["confidenceMetricHeartOut_xCorr"] = confidenceMetricHeartOut_xCorr
        new_data["sumEnergyBreathWfm"] = sumEnergyBreathWfm
        new_data["sumEnergyHeartWfm"] = sumEnergyHeartWfm
        new_data["rsv_0"] = rsv_0
        new_data["rsv_1"] = rsv_1
        """
        # ------------- Discard the first 60 seconds -------------
        new_data.drop(new_data.index[0:60*5], inplace=True) 
        # print(os.path.join("sleep_features", datas))
        # new_data.to_csv(os.path.join("sleep_features", datas), index=False)
        print(os.path.join("sleep_features", datas))
        new_data.to_csv(os.path.join("sleep_features", datas), index=False)
        print("Completed!")