import pandas as pd
import numpy as np
import os

# 秒轉分鐘
def sec2min(data):
    heart_ar = []
    breath_ar = []
    sleep_ar = []
    tfRSA_ar = []
    tmHR_ar = []
    sfRSA_ar = []
    smHR_ar = []
    sdfRSA_ar = []
    sdmHR_ar = []
    stfRSA_ar = []
    stmHR_ar = []
    datetime_ar = []

    next_HM = False
    strat_index = 0
    loc_time = data.datetime
    start_hour = loc_time[0][:2]
    start_min = loc_time[0][3:5]

    for loc in range(len(loc_time)):
        end_hour = loc_time[loc][:2]
        end_min = loc_time[loc][3:5]

        # 小時
        if int(end_hour) - int(start_hour) >= 1:
            start_hour = end_hour
            start_min = end_min
            next_HM = True

        # 分鐘
        if int(end_min) - int(start_min) >= 1:
            start_min = end_min
            next_HM = True  
      
        if next_HM:
            end_index = int(loc - 1)
            datetime_ar.append(end_hour + ":" + end_min + ":00")
            heart_ar.append(np.mean(data["heart"][strat_index:end_index]))
            breath_ar.append(np.mean(data["breath"][strat_index:end_index]))
            sleep_ar.append(np.mean(data["sleep"][strat_index:end_index]))
            tfRSA_ar.append(np.mean(data["tfRSA"][strat_index:end_index]))
            tmHR_ar.append(np.mean(data["tmHR"][strat_index:end_index]))
            sfRSA_ar.append(np.mean(data["sfRSA"][strat_index:end_index]))
            smHR_ar.append(np.mean(data["smHR"][strat_index:end_index]))
            sdfRSA_ar.append(np.mean(data["sdfRSA"][strat_index:end_index]))
            sdmHR_ar.append(np.mean(data["sdmHR"][strat_index:end_index]))
            stfRSA_ar.append(np.mean(data["stfRSA"][strat_index:end_index]))
            stmHR_ar.append(np.mean(data["stmHR"][strat_index:end_index]))
            next_HM = False
            strat_index = int(loc)

    data_min_tmp = {"datetime":datetime_ar, "heart":heart_ar, "breath":breath_ar, "sleep":sleep_ar, "tfRSA":tfRSA_ar, "tmHR":tmHR_ar, "sfRSA":sfRSA_ar, "smHR":smHR_ar, "sdfRSA":sdfRSA_ar, "sdmHR":sdmHR_ar, "stfRSA":stfRSA_ar, "stmHR":stmHR_ar}
    data_min = pd.DataFrame(data_min_tmp)
    return data_min

if __name__ == "__main__":
    data_path = "./sleep_features/"
    for name in os.listdir(data_path):
        loc_data = os.path.join(data_path, name)
        data = pd.read_csv(loc_data)
        data_min = sec2min(data)
        print(f"Now: {name}")
        data_min.to_csv("./sleep_features_min/" + name, index=False)

