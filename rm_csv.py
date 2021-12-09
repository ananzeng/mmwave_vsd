import pandas as pd
import csv
import os
os.mkdir(os.path.join("dataset_sleep_test", "processed_data"))
for name in os.listdir(os.path.join("dataset_sleep_test")):
    print("正在處理：", name)
    df = pd.read_csv(os.path.join("dataset_sleep_test", name))

    for i in range(df.shape[0]-1500, df.shape[0], 1):
        second = df['datetime'][i][-2:]
        #print(second)
        if  second == "00":
            #print(i)
            break
    df.drop(df.index[i:df.shape[0]],inplace=True) #刪除1,2行的整行數據

    for i in range(1500):
        second = df['datetime'][i][-2:]
        #print(second)
        if  second == "00":
            #print(i)
            break
    df.drop(df.index[0:i],inplace=True) #刪除1,2行的整行數據


    df.to_csv(os.path.join("dataset_sleep_test", "processed_data", name[:-4]+ "_processed_data.csv"),index=False,encoding="utf-8")