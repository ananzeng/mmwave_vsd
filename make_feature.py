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

names = "./dataset_sleep_test"
raw_data_pd = pd.DataFrame()

files = os.path.join(names, "processed_data")
print(files)
for num in range(len(os.listdir(files))):
    datas = os.listdir(files)[num]
    data = os.path.join(files, datas)
    print(data)
    raw_data_pd = pd.read_csv(data)


    raw_data_pd['heart'] = raw_data_pd['heart'].replace(0, np.nan)
    new_data = raw_data_pd.dropna()
    breath = np.array(new_data['breath'])
    heart = np.array(new_data['heart'])
    print(f"Total len: {len(heart)}\n")

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
    new_data.insert(39, "tfRSA", local_tfRSA)
    new_data.insert(40, "tmHR", local_tmHR)
    new_data.insert(41, "sfRSA", local_sfRSA)
    new_data.insert(42, "smHR", local_smHR)
    new_data.insert(43, "sdfRSA", local_sdfRSA)
    new_data.insert(44, "sdmHR", local_sdmHR)

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

    new_data.insert(45, "stfRSA", local_stfRSA)
    new_data.insert(46, "stmHR", local_stmHR)

    print("\nsfRSA and smHR Complete!")
    print(f"Real len: {len(heart)}\nstfRSA len: {len(local_stfRSA)}\nstmHR len: {len(local_stmHR)}")

    all_time = []
    datetime_array = np.array(new_data['datetime'])
    time_800 = datetime.datetime.strptime('20:00:00', "%H:%M:%S")
    for i in datetime_array:
        #print(i)
        i = datetime.datetime.strptime(str(i), "%H:%M:%S")
        time_feature = i - time_800
        #print(time_feature.seconds)
        all_time.append(time_feature.seconds)
    new_data.insert(47, "time", all_time)

    print("\ntime Complete!")
    print(f"Real len: {len(heart)}\ntime len: {len(all_time)}")
    #-----------------------------------
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
    new_data.insert(48, "sleep_counter", stage_counter)
    
    print("\ntime Complete!")
    print(f"Real len: {len(heart)}\ntime len: {len(all_time)}")

    new_data.drop(new_data.index[0:60],inplace=True) 
    print(os.path.join("sleep_features", datas))
    new_data.to_csv(os.path.join("sleep_features", datas), index=False)
    print("Completed!")