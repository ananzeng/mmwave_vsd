import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
user = "1209"
files_path = os.path.join("dataset_sleep", user, "0.8")
files = os.listdir(files_path)    
for tmp in range(0, len(files)//2, 1):
    file = files[tmp]
    print(f'\nCurrent file: {file}')
    datas_path = os.path.join(files_path, file)
    vitial_sig = pd.read_csv(datas_path)
    all_time = vitial_sig['datetime'].values
    all_heart = vitial_sig['heart'].values
    all_breath = vitial_sig['breath'].values
    all_sleep_stage = vitial_sig['sleep'].values
    temp_all_time = []
    temp_all_heart = []
    temp_all_breath = []

    time = []
    heart = []
    breath = []
    sleep = []
    for i in range(0, len(all_time)-1):
        if all_time[i] == all_time[i+1]:
            temp_all_time.append(all_time[i])
            temp_all_heart.append(all_heart[i])
            temp_all_breath.append(all_breath[i])
        else:
            temp_all_time.append(all_time[i])
            temp_all_heart.append(all_heart[i])
            temp_all_breath.append(all_breath[i])
            if sorted(temp_all_time)[0] == sorted(temp_all_time)[-1]:
                print(temp_all_time[0])
                time.append(temp_all_time[0])
            else:
                print("warn")
                break     
            index = np.where(np.array(temp_all_heart) > 0)
            one = np.array(index)
            if one.size == 1:
                heart.append(temp_all_heart[int(one[0])])
                breath.append(temp_all_breath[int(one[0])])      
                sleep.append(all_sleep_stage[i])
            elif one.size == 0:
                print("全0")
                heart.append(0)
                breath.append(0)   
                sleep.append(0)
            else:
                print("warn")
                break              
            print(temp_all_breath)   
            temp_all_time = []
            temp_all_heart = []
            temp_all_breath = []
    plt.figure(figsize=(12,4))
    plt.subplot(311)
    plt.title('heart rate')

    plt.plot(np.arange(len(time)) ,heart, "b", linewidth = 1)
    plt.subplot(312)
    plt.title('breath rate')

    plt.plot(np.arange(len(time)) ,breath, "g")
    plt.subplot(313)
    plt.title('sleep stage  5=Wake, 4=REM, 3=Light Sleep, 2=Deep Sleep')

    plt.plot(np.arange(len(time)) ,sleep, "y")
    plt.show()
    