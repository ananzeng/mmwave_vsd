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
rf_p = []
def load_file(path):
    heart = []
    breath = []
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

    return heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, sleep, time, sleep_counter



heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, sleep, time, sleep_counter = load_file(os.path.join("sleep_features_sec", "train"))
X_train = [heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, time, sleep_counter]
y_train = sleep
all_data = np.array(X_train).transpose()
all_gt_array = np.array(y_train)

rf = RandomForestClassifier(n_estimators = 8, random_state = 69, n_jobs = -1, min_samples_leaf = 3, min_samples_split = 5)
rf.fit(all_data, all_gt_array)
joblib.dump(rf, "save/sleep_feature_sec_rf.pkl")
clf_rbf = svm.SVC(random_state = 69, C=7, kernel='rbf', degree=5, gamma='scale')
clf_rbf.fit(all_data, all_gt_array) 
joblib.dump(clf_rbf, "save/sleep_feature_sec_svm.pkl")
neigh = KNeighborsClassifier(n_neighbors = 17, n_jobs = -1)
neigh.fit(all_data, all_gt_array) 
joblib.dump(neigh, "save/sleep_feature_sec_knn.pkl")
xgbrmodel = xgb.XGBClassifier(n_estimators = 8, random_state = 69, n_jobs = -1)
xgbrmodel.fit(all_data, all_gt_array) 
joblib.dump(xgbrmodel, "save/sleep_feature_sec_xgb.pkl")


rf2 = joblib.load('save/sleep_feature_sec_rf.pkl')
clf2 = joblib.load('save/sleep_feature_sec_svm.pkl')
neigh2 = joblib.load('save/sleep_feature_sec_knn.pkl')
xgbrmodel2 = joblib.load('save/sleep_feature_sec_xgb.pkl')

heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR , sleep, time, sleep_counter = load_file(os.path.join("sleep_features_sec", "test"))
X_test = [heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, time, sleep_counter]
y_test = sleep
all_data = np.array(X_test).transpose()
all_gt_array = np.array(y_test)


'''
def final_windows(ans_ar):
    from scipy import stats
    window_size = 5
    for index in range(2 * (len(ans_ar) // window_size) + 1):
        windows = ans_ar[index*window_size//2:(window_size//2 + index*window_size//2)]
        for tmp in range(window_size//2):
            ans_ar[index*window_size//2:(tmp + index  *window_size//2)] = windows[np.argmax(windows)]
    return ans_ar
rf_ans =   final_windows(rf2.predict(all_data))
svm_ans =   final_windows(clf2.predict(all_data))
knn_ans =   final_windows(neigh2.predict(all_data))
xgb_ans =   final_windows(xgbrmodel2.predict(all_data))
'''
print("RF", accuracy_score(rf2.predict(all_data), all_gt_array))
print("SVM", accuracy_score(clf2.predict(all_data), all_gt_array))
print("KNN", accuracy_score(neigh2.predict(all_data), all_gt_array))
print("XGB RF", accuracy_score(xgbrmodel2.predict(all_data), all_gt_array))
print()
'''
print("RF", accuracy_score(rf_ans, all_gt_array))
print("SVM", accuracy_score(svm_ans, all_gt_array))
print("KNN", accuracy_score(knn_ans, all_gt_array))
print("XGB RF", accuracy_score(xgb_ans, all_gt_array))
'''
plt.figure(figsize=(12,4))
plt.subplot(311)
plt.plot(np.arange(len(all_gt_array)) ,rf2.predict(all_data), "b")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('rf')

plt.subplot(312)
plt.plot(np.arange(len(all_gt_array)) ,neigh2.predict(all_data), "g")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('knn')

plt.subplot(313)
plt.plot(np.arange(len(all_gt_array)) ,xgbrmodel2.predict(all_data), "y")
plt.plot(np.arange(len(all_gt_array)) ,all_gt_array, "r")
plt.title('xbg_rf')
plt.show()
