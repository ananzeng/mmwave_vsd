import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score
import xgboost as xgb
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
    sleep_feature_path = path
    for i in os.listdir(sleep_feature_path):
        print("現在檔案：", i)
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
    return heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR, sleep
heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR , sleep = load_file(os.path.join("sleep_features_min", "train"))
X_train = [heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR]
y_train = sleep
all_data = np.array(X_train).transpose()
all_gt_array = np.array(y_train)
rf = RandomForestClassifier()
rf.fit(all_data, all_gt_array)
joblib.dump(rf, "save/sleep_feature_min_rf.pkl")
clf_rbf = svm.SVC()
clf_rbf.fit(all_data, all_gt_array) 
joblib.dump(clf_rbf, "save/sleep_feature_min_svm.pkl")
neigh = KNeighborsClassifier()
neigh.fit(all_data, all_gt_array) 
joblib.dump(neigh, "save/sleep_feature_min_knn.pkl")
xgbrmodel = xgb.XGBClassifier()
xgbrmodel.fit(all_data, all_gt_array) 
joblib.dump(xgbrmodel, "save/sleep_feature_min_xgb.pkl")

rf2 = joblib.load('save/sleep_feature_min_rf.pkl')
clf2 = joblib.load('save/sleep_feature_min_svm.pkl')
neigh2 = joblib.load('save/sleep_feature_min_knn.pkl')
xgbrmodel2 = joblib.load('save/sleep_feature_min_xgb.pkl')

heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR , sleep = load_file(os.path.join("sleep_features_min", "test"))
X_test = [heart, breath, tfRSA, tmHR, sfRSA, smHR, sdfRSA, sdmHR, stfRSA, stmHR]
y_test = sleep
all_data = np.array(X_test).transpose()
all_gt_array = np.array(y_test)
print(accuracy_score(rf2.predict(all_data), all_gt_array))
print(accuracy_score(clf2.predict(all_data), all_gt_array))
print(accuracy_score(neigh2.predict(all_data), all_gt_array))
print(accuracy_score(xgbrmodel2.predict(all_data), all_gt_array))