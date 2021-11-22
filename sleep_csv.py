import os
import pandas as pd
import csv
import numpy as np
path = os.path.join("dataset_sleep", "1110test", "0.8", "0.csv")
stage_csv = os.path.join("dataset_sleep", "1110test", "0.8", "stage.csv")
stage_csv_data = os.path.join("dataset_sleep", "1110test", "0.8", "data.csv")
print(path)
start = True
end = True
temp = []
stage = ["00:23:49", "00:30:00", "00:30:00", "00:33:00", "00:33:00",  "00:43:00", "00:43:00", "00:55:00"]
sleep_stage = ["5", "3", "2", "3"]

vitial_sig = pd.read_csv(stage_csv_data)
sleep_stage = vitial_sig['sleep_stage'].values
sleep_stage = sleep_stage.tolist()
sleep_stage.append(5)
start = vitial_sig['start'].values
end = vitial_sig['end'].values

vitial_sig_1 = pd.read_csv(path)
total = len(vitial_sig_1['heart'].values)

stage = []
for i in range(len(start)):
  stage.append(str(start[i]))
  stage.append(str(end[i]))
print("sleep_stage：", sleep_stage)
print("stage", stage)

for i in range(0, len(stage), 2):
  print(i)
  start = True
  end = True
  with open(path, newline='') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    # 以迴圈輸出每一列
    for index, row in enumerate(rows):
      #print(index)
      #print(row[35])
      if row[35] == str(stage[i]): #start
        if start:
          temp.append(index + 1)     
          start = False  
      if row[35] == str(stage[i + 1]): #end
        if end:
          temp.append(index) 
          end = False  
          break
    #print(temp)
temp[0] = temp[0] - 1 
print("total", total)
temp.append(temp[-1] + 1)
temp.append(total)
print(temp)
with open(stage_csv, 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
  writer = csv.writer(csvfile)
  #writer.writerow(["sleep_stage"])
  # 寫入一列資料
  for i in range(0, len(temp), 2):
    print(i//2)
    for k in range(int(temp[i+1]) - int(temp[i]) + 1):
      writer.writerow(str(sleep_stage[i//2]))

print("整段複製 刪掉第一行")