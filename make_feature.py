import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import combine_svm
import pandas as pd
from tqdm import tqdm
import csv
import scipy.signal
from scipy.fftpack import fft
import seaborn as sns
import datetime,time

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

    # ------------------------ Drop ------------------------ 
    raw_data_pd['heart'] = raw_data_pd['heart'].replace(0, np.nan)
    new_data = raw_data_pd.dropna()
    breath = np.array(new_data['breath'])
    heart = np.array(new_data['heart'])
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

    # ------------- Discard the first 60 seconds -------------
    new_data.drop(new_data.index[0:60*5], inplace=True) 
    # print(os.path.join("sleep_features", datas))
    # new_data.to_csv(os.path.join("sleep_features", datas), index=False)
    print(os.path.join("sleep_features", datas))
    new_data.to_csv(os.path.join("sleep_features", datas), index=False)
    print("Completed!")