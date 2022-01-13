import os
import pandas as pd
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
path = os.path.join("dataset_sleep", "0109_mi_6_test_pohua", "0.8", "0.csv")
stage_csv_mi_band_6 = os.path.join("dataset_sleep", "0109_mi_6_test_pohua", "0.8", "stage_mi_band_6.csv")
stage_csv_mi_watch = os.path.join("dataset_sleep", "0109_mi_6_test_pohua", "0.8", "stage_mi_watch.csv")
print(path)
vitial_sig = pd.read_csv(stage_csv_mi_band_6)
mi_band_6_sleep_stage = vitial_sig['3'].values

vitial_sig = pd.read_csv(stage_csv_mi_watch)
mi_watch_sleep_stage = vitial_sig['2'].values
result = accuracy_score(mi_watch_sleep_stage, mi_band_6_sleep_stage)
print("result", result)

plt.figure(figsize=(12,4))
plt.subplot(211)
plt.title('mi_band_6')
plt.xticks([])
plt.plot(np.arange(len(mi_band_6_sleep_stage)) ,mi_band_6_sleep_stage, "b", linewidth = 1)

plt.subplot(212)
plt.title('mi_watch')
plt.plot(np.arange(len(mi_watch_sleep_stage)) ,mi_watch_sleep_stage, "g")
plt.savefig("0109_pohua_"+ str(result) + ".png")
plt.show()
