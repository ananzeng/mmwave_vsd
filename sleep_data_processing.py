import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
user = "1110test"
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
    temp_all_time = []
    temp_all_heart = []
    temp_all_breath = []

    time = []
    heart = []
    breath = []

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
            elif one.size == 0:
                print("全0")
                heart.append(0)
                breath.append(0)   
            else:
                print("warn")
                break              
            print(temp_all_breath)   
            temp_all_time = []
            temp_all_heart = []
            temp_all_breath = []

    plt.plot(np.arange(len(time)),heart)
    plt.plot(np.arange(len(time)),breath)
    plt.show()
    