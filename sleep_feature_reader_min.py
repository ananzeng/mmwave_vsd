import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import preprocessing

rf_p = []
def load_file(path):
    heart = []
    breath = []
    mov_dens = []
    LF = []
    HF = []
    LFHF = []
    sHF = []
    sLFHF = []
    tfRSA = []
    tmHR = []
    sfRSA = []
    smHR = []
    sdfRSA = []
    sdmHR = []
    stfRSA = []
    stmHR = []
    sleep = []
    time = []
    sleep_counter = []
    sleep_feature_path = path
    for i in os.listdir(sleep_feature_path):
        #print("現在檔案：", i)
        sleep_features = pd.read_csv(os.path.join(sleep_feature_path, i))
        for number in range(sleep_features.shape[0]):
            heart.append(sleep_features["heart"].values[number])
            breath.append(sleep_features["breath"].values[number])
            mov_dens.append(sleep_features["mov_dens"].values[number])
            LF.append(sleep_features["LF"].values[number])
            HF.append(sleep_features["HF"].values[number])
            LFHF.append(sleep_features["LFHF"].values[number])
            sHF.append(sleep_features["sHF"].values[number])
            sLFHF.append(sleep_features["sLFHF"].values[number])
            tfRSA.append(sleep_features["tfRSA"].values[number])
            tmHR.append(sleep_features["tmHR"].values[number])
            sfRSA.append(sleep_features["sfRSA"].values[number])
            smHR.append(sleep_features["smHR"].values[number])
            sdfRSA.append(sleep_features["sdfRSA"].values[number])
            sdmHR.append(sleep_features["sdmHR"].values[number])
            stfRSA.append(sleep_features["stfRSA"].values[number])
            stmHR.append(sleep_features["stmHR"].values[number])
            sleep.append(sleep_features["sleep"].values[number])
            time.append(sleep_features["time"].values[number])
            sleep_counter.append(sleep_features["sleep_counter"].values[number])

    return heart, breath, mov_dens, LF, HF, LFHF, sHF, sLFHF, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, sleep, time, sleep_counter

def label_analyze(label):
    array_5 = 0
    array_4 = 0
    array_3 = 0
    array_2 = 0
    for i in label:
        if i == 5:
            array_5 += 1
        elif i == 4:
            array_4 += 1
        elif i == 3:
            array_3 += 1
        elif i == 2:
            array_2 += 1
    print("睡眠階段統計 清醒", array_5, "分鐘 快速動眼期", array_4, "分鐘 淺眠", array_3, "分鐘 深眠", array_2, "分鐘")

def final_windows(ans_ar):
    from scipy import stats
    window_size = 3
    for index in range(2 * (len(ans_ar) // window_size) + 1):
        windows = ans_ar[index*window_size//2:(window_size//2 + index*window_size//2)]
        for tmp in range(window_size//2):
            ans_ar[index*window_size//2:(tmp + index  *window_size//2)] = windows[np.argmax(windows)]
    return ans_ar


heart, breath, mov_dens, LF, HF, LFHF, sHF, sLFHF, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, sleep, time, sleep_counter = load_file(os.path.join("sleep_features_min", "train"))
X_train = [heart, breath, mov_dens, LF, HF, LFHF, sHF, sLFHF, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, time, sleep_counter]
y_train = sleep
all_data = np.array(X_train).transpose()
all_gt_array = np.array(y_train)
print("train data 是否有nan:", np.isnan(all_data.any()))
print("train label 是否有nan:", np.isnan(all_gt_array.any()))
all_data = all_data.astype('float64')
all_gt_array = all_gt_array.astype('float64')
label_analyze(all_gt_array)
rf = RandomForestClassifier(n_estimators = 600, random_state = 69, n_jobs = -1)
rf.fit(all_data, all_gt_array)
joblib.dump(rf, "save/sleep_feature_min_rf.pkl")
clf_rbf = svm.SVC(random_state = 69)
clf_rbf.fit(all_data, all_gt_array) 
joblib.dump(clf_rbf, "save/sleep_feature_min_svm.pkl")
neigh = KNeighborsClassifier(n_neighbors=17, n_jobs = -1)
neigh.fit(all_data, all_gt_array) 
joblib.dump(neigh, "save/sleep_feature_min_knn.pkl")
xgbrmodel = xgb.XGBClassifier(n_estimators = 600, random_state = 69, n_jobs = -1)
xgbrmodel.fit(all_data, all_gt_array) 
joblib.dump(xgbrmodel, "save/sleep_feature_min_xgb.pkl")

#test

rf2 = joblib.load('save/sleep_feature_min_rf.pkl')
clf2 = joblib.load('save/sleep_feature_min_svm.pkl')
neigh2 = joblib.load('save/sleep_feature_min_knn.pkl')
xgbrmodel2 = joblib.load('save/sleep_feature_min_xgb.pkl')

heart, breath, mov_dens, LF, HF, LFHF, sHF, sLFHF, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR , sleep, time, sleep_counter = load_file(os.path.join("sleep_features_min", "test"))
X_test = [heart, breath, mov_dens, LF, HF, LFHF, sHF, sLFHF, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, time, sleep_counter]

y_test = sleep
all_data = np.array(X_test).transpose()
all_gt_array = np.array(y_test)
label_analyze(all_gt_array)
print("test data 是否有nan:", np.isnan(all_data.any()))
print("test label 是否有nan:", np.isnan(all_gt_array.any()))
print("RF", accuracy_score(rf2.predict(all_data), all_gt_array))
print("SVM", accuracy_score(clf2.predict(all_data), all_gt_array))
print("KNN", accuracy_score(neigh2.predict(all_data), all_gt_array))
print("XGB RF", accuracy_score(xgbrmodel2.predict(all_data), all_gt_array))

rf_ans =   final_windows(rf2.predict(all_data))
svm_ans =   final_windows(clf2.predict(all_data))
knn_ans =   final_windows(neigh2.predict(all_data))
xgb_ans =   final_windows(xgbrmodel2.predict(all_data))
print()
print("RF", accuracy_score(rf_ans, all_gt_array))
print("SVM", accuracy_score(svm_ans, all_gt_array))
print("KNN", accuracy_score(knn_ans, all_gt_array))
print("XGB RF", accuracy_score(xgb_ans, all_gt_array))

#plt.figure(figsize=(12,4))
'''
plt.subplot(411)
plt.plot(np.arange(len(all_gt_array)) ,rf2.predict(all_data), "b")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('rf')

plt.subplot(412)
plt.plot(np.arange(len(all_gt_array)) ,clf2.predict(all_data), "b")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('svm')

plt.subplot(413)
plt.plot(np.arange(len(all_gt_array)) ,neigh2.predict(all_data), "g")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('knn')

plt.subplot(414)
'''
plt.plot(np.arange(len(all_gt_array)) ,xgbrmodel2.predict(all_data), "y", label='predict')
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r", label='ground truth')
plt.title('rf')
plt.legend(loc='upper left', fontsize=20)
plt.xlabel('mins', fontsize=20)
plt.ylabel('sleep stage', fontsize=20)
plt.show()
