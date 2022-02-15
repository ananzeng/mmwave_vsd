import os
import csv
import serial
import pickle
import joblib
import datetime, time

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.brhr_function import substitute, detect_Breath
from utils.sleep_stage import *
from utils.util import *

import vitalsign_v2


if __name__ == "__main__":
    count = 0
    # change = True
    begin = False
    coco = True
    switch = True
    next_YMD = False
    open_stfRSA = False
    open_stmHR = False
    open_HF = False
    parameters = np.array([[1.5, 0.125, 0.55, 20, 5, 2, 22, 17], [1.5, 0.9, 1.9, 20, 9, 2, 5, 4]])

    # Data location
    path = "./dataset_sleep/tester/"
    make_file(path)
    data_number = str(num_data(path))
    path_data = path + data_number +".csv"
    path_range_bin = path + "Range_bins_"+ data_number +".csv"

    # Write csv column name
    with open(path_data, "a", newline="") as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(["heart", "breath", "bmi", "deep_p", "ada_br", "ada_hr", "var_RPM", "var_HPM", "rem_parameter", "mov_dens", "LF", "HF", "LFHF",
                	    "sHF", "sLFHF", "tfRSA", "tmHR", "sfRSA", "smHR", "sdfRSA", "sdmHR", "stfRSA", "stmHR", "time", "datetime", "sleep"])
    with open(path_range_bin, "a",newline="") as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(["rangeBinInde"])

    print(" ----------- Start in 5 seconds ----------- ")
    # time.sleep(5)

    # Data initial
    port = serial.Serial("COM5", baudrate = 921600, timeout = 0.5)
    vts = vitalsign_v2.VitalSign(port)

    # 載入模型
    rf2 = joblib.load('save/sleep_feature_min_rf.pkl')

    raw_sig = []  # 訊號的窗格
    energe_br = []  # 呼吸能量的窗格
    energe_hr = []  # 心跳能量的窗格
    heart_ti = []  # TI心跳
    breath_ti = []  # TI呼吸
    tmp_br = 0  # 初始化前一秒呼吸律
    tmp_hr = 0  # 初始化前一秒心律
    tfRSA_arr = []
    tmHR_arr = []
    HF_arr = []
    LFHF_arr = []

    # KNN features
    counter = 0
    raw_sig_KNN = []
    var_RPM_br_KNN = []
    var_RPM_hr_KNN = []

    # 秒 => 分鐘
    counter_mean = 0
    heart_ar = []
    breath_ar = []
    sleep_ar = []
    bmi_ar = []
    deep_p_ar = []
    ada_br_ar = []
    ada_hr_ar = []
    var_RPM_ar = []
    var_HPM_ar= []
    rem_parameter_ar = []
    mov_dens_ar = []
    LF_ar = []
    HF_ar = []
    LFHF_ar = []
    sHF_ar = []
    sLFHF_ar = []
    tfRSA_ar = []
    tmHR_ar = []
    sfRSA_ar = []
    smHR_ar = []
    sdfRSA_ar = []
    sdmHR_ar = []
    stfRSA_ar = []
    stmHR_ar = []
    time_ar = []
    next_HM = False

    # 計算前幾分鐘
    looper = 0
    tmp_rest = []

    # final_results
    all_results = np.zeros(24)
    port.flushInput() # 丟棄接收緩存中的所有數據
    time_Start = time.time()
    
    # 每秒算一次
    ct = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") # 時間格式為字串
    start_year = ct[0:4]  
    start_month = ct[5:7]
    start_day = ct[8:10]
    start_time = int(ct[11:13])*3600 + int(ct[14:16])*60 + int(ct[17:19])

    # Start recording
    while True:
        (dck , vd, rangeBuf) = vts.tlv_read(False)
        vs = vts.getHeader()
        if dck:
            raw_sig.append(vd.unwrapPhasePeak_mm)
            raw_sig_KNN = list(np.copy(raw_sig))
            heartRateEst_FFT_mean = np.mean(vd.heartRateEst_FFT)
            heartRateEst_xCorr_mean = np.mean(vd.heartRateEst_xCorr)
            heart_ti.append(vd.rsv[1])
            breath_ti.append(vd.rsv[0])
            hr_rate = 0
            br_rate = 0
            time_End = time.time()

            # Time 40 seconds
            if coco:
                print(f"Elapsed time (sec): {round(time_End - time_Start, 3)}")
            
            if len(raw_sig) > 40*20:
                coco = False
                try:
                    # 只取最後 40 秒的值
                    current_window_sig = raw_sig[-40*20:]
                    current_heart_ti = heart_ti[-40*20:]
                    heart_ti.pop(0)
                    current_breath_ti = breath_ti[-40*20:]
                    breath_ti.pop(0)

                    # KNN features
                    current_window_bmi = raw_sig[-60*20:]

                    # LF_HF_LFHF features
                    LF_HF_LFHF_windows = raw_sig[-5*60*20:]
                    if len(LF_HF_LFHF_windows) >= 5*60*20:
                        raw_sig.pop(0)


                    next_YMD, start_time, end_time, sec = define_time(next_YMD, start_year, start_month, start_day, start_time)
                    if (end_time - start_time >= 1) or next_YMD == True:
                        next_YMD == False

                        # 呼吸
                        br_rate, index_of_fftmax = detect_Breath(current_window_sig, parameters[0][:])
                        with open("save/svm_br_office_all.pickle", "rb") as f:
                            clf = pickle.load(f)
                            svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                            if svm_predict == 1:
                                br_rate = np.mean(current_breath_ti)

                        # 心跳
                        hr_rate ,index_of_fftmax = detect_Breath(current_window_sig, parameters[1][:])
                        with open("save/svm_hr_office_all.pickle", "rb") as f:
                            clf = pickle.load(f)
                            svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                            if svm_predict == 1:
                                hr_rate = np.mean(current_heart_ti)

                        br_rpm = br_rate
                        hr_rpm = hr_rate
                        br_rpm = substitute(tmp_br, br_rpm, 1)
                        hr_rpm = substitute(tmp_hr, hr_rpm, 0)
                        br_rpm = np.round(br_rpm, 4)
                        hr_rpm = np.round(hr_rpm, 4)

                        if sec == "00" and counter == 0:
                            counter += 1
                            begin = True
                            print("開始建立資料")
                        
                        # 滿足秒數為 00
                        if begin:
                            var_RPM_br_KNN.append(br_rpm)
                            var_RPM_hr_KNN.append(hr_rpm)
                            """ 睡眠階段 (Paper1) """
                            """ 動作 """
                            # mov_dens
                            if len(current_window_bmi) == 60 * 20:  # 60(秒) 20(取樣頻率)
                                mov_dens = mov_dens_fn(current_window_bmi)
                                print(f"mov_dens: {mov_dens}")
                                mov_dens_ar.append(mov_dens)

                            """ 呼吸 """
                            # tfRSA
                            if len(var_RPM_br_KNN) >= 10:  # 原本為 10, 窗格改為20
                                tfRSA = tfRSA_fn(var_RPM_br_KNN[-10:])
                                tfRSA_arr.append(tfRSA)
                                print(f"tfRSA: {tfRSA}")
                                tfRSA_ar.append(tfRSA)
                                if len(tfRSA_arr) >= 31:
                                    open_stfRSA = True

                            # sfRSA
                            if len(var_RPM_br_KNN) >= 31:
                                sfRSA, sfRSA_mean = sfRSA_fn(var_RPM_br_KNN[-31:])
                                print(f"sfRSA: {sfRSA_mean}")
                                sfRSA_ar.append(sfRSA_mean)

                            # stfRSA
                            if open_stfRSA:
                                stfRSA, stfRSA_mean = stfRSA_fn(tfRSA_arr)
                                print(f"stfRSA: {stfRSA_mean}")
                                stfRSA_ar.append(stfRSA_mean)
                                tfRSA_arr.pop(0)
                            
                            # sdfRSA
                            if len(var_RPM_br_KNN) >= 31:
                                sdfRSA, sdfRSA_mean = sdfRSA_fn(var_RPM_br_KNN[-31:], sfRSA)
                                print(f"sdfRSA: {sdfRSA_mean}")
                                sdfRSA_ar.append(sdfRSA_mean)
                            
                            """ 心跳 """
                            # tmHR
                            if len(var_RPM_hr_KNN) >= 10:  # 原本為 10, 窗格改為20
                                tmHR = tmHR_fn(var_RPM_hr_KNN[-10:])
                                tmHR_ar.append(tmHR)
                                tmHR_arr.append(tmHR)
                                if len(tmHR_arr) >= 31:
                                    open_stmHR = True

                            # smHR
                            if len(var_RPM_hr_KNN) >= 31:
                                smHR, smHR_mean = smHR_fn(var_RPM_hr_KNN[-31:])
                                smHR_ar.append(smHR_mean)

                            # stmHR
                            if open_stmHR:
                                stmHR, stmHR_mean = stmHR_fn(tmHR_arr)
                                stmHR_ar.append(stmHR_mean)
                                tmHR_arr.pop(0)
                            
                            # sdmHR
                            if len(var_RPM_hr_KNN) >= 31:
                                sdmHR, sdmHR_mean = sdmHR_fn(var_RPM_hr_KNN[-31:], smHR)
                                sdmHR_ar.append(sdmHR_mean)

                            # LF_HF_LFHF
                            LF, HF, LFHF = LF_HF_LFHF(LF_HF_LFHF_windows)
                            print(f"LF: {LF}")
                            print(f"HF: {HF}")
                            print(f"LFHF: {LFHF}")
                            LF_ar.append(LF)
                            HF_ar.append(HF)
                            LFHF_ar.append(LFHF)
                            HF_arr.append(HF)
                            LFHF_arr.append(LFHF)
                            if len(HF_arr) >= 31:
                                open_HF = True

                            if open_HF:
                                # sHF
                                sHF, sHF_mean = sHF_fn(HF_arr)
                                # sLFHF
                                sLFHF, sLFHF_mean = sLFHF_fn(LFHF_arr)
                                print(f"sHF: {sHF_mean}")
                                print(f"sLFHF: {sLFHF_mean}")
                                sHF_ar.append(sHF_mean)
                                sLFHF_ar.append(sLFHF_mean)
                                HF_arr.pop(0)
                                LFHF_arr.pop(0)

                            """ 睡眠階段 (Paper2) """
                            # Variance of RPM
                            if len(var_RPM_br_KNN) == 10 * 60:
                                var_RPM_br = var_RPM(var_RPM_br_KNN)
                                var_RPM_hr = var_RPM(var_RPM_hr_KNN)
                                var_RPM_br_KNN.pop(0)
                                var_RPM_hr_KNN.pop(0)
                                var_RPM_ar.append(var_RPM_br)
                                var_HPM_ar.append(var_RPM_hr)
                                
                                # 進入迴圈後，前 10 分鐘不紀錄
                                looper += 1
                                print(f"Variance of RPM: BR = {var_RPM_br}, HR = {var_RPM_hr}")
                            
                            # Body Movement Index (BMI)
                            # Deep Parameter
                            if len(current_window_bmi) == 60 * 20:
                                bmi_current = bmi(current_window_bmi)
                                hk = np.mean(var_RPM_hr_KNN[-60:])
                                dk = deep_parameter(bmi_current, hk)
                                print(f"Body Movement Index: {bmi_current}")
                                bmi_ar.append(bmi_current)
                                print(f"Deep Parameter: {dk}")
                                deep_p_ar.append(dk)
                                # Amplitude Difference Accumulation (ADA) of Respiration
                                ada_br = ada(current_window_bmi, brhr = 0)
                                ada_hr = ada(current_window_bmi, brhr = 1)
                                print(f"ADA: BR = {ada_br}, HR = {ada_hr}")
                                ada_br_ar.append(ada_br)
                                ada_hr_ar.append(ada_hr)

                            # REM Parameter
                            if len(var_RPM_br_KNN) >= 5 * 60:
                                rem_p = rem_parameter(var_RPM_br_KNN[-5*60:])
                                print(f"REM Parameter: {rem_p}") 
                                rem_parameter_ar.append(rem_p)

                            # Time features
                            ct3 = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") # 時間格式為字串
                            tmp_time = time_fn(ct3[11:19])
                            print(f"Time features: {tmp_time}")
                            time_ar.append(tmp_time)
                            breath_ar.append(br_rpm)
                            heart_ar.append(hr_rpm)

                            # print("TIME：", ct2)
                            print(f"Breathe Rate per minute: {br_rpm}")
                            print(f"Heart Rate per minute: {hr_rpm}")
                            print(f"Len of br: {len(var_RPM_br_KNN)}")
                            print()
                            
                            if counter_mean == 0:
                                loc_time = ct3[11:19]
                                start_hour = loc_time[:2]
                                start_min = loc_time[3:5]
                                end_hour = start_hour
                                end_min = start_min
                                counter_mean += 1
                            else:
                                loc_time = ct3[11:19]
                                end_hour = loc_time[:2]
                                end_min = loc_time[3:5]

                            # 小時
                            if int(end_hour) - int(start_hour) >= 1 or int(end_hour) - int(start_hour) == -23:
                                start_hour = end_hour
                                start_min = end_min
                                next_HM = True

                            # 分鐘
                            if int(end_min) - int(start_min) >= 1:
                                start_min = end_min
                                next_HM = True  
                        
                            if next_HM and looper >= 10:
                                all_results[22] = np.mean(breath_ar)
                                all_results[23] = np.mean(heart_ar)
                                all_results[0] = np.mean(bmi_ar)
                                all_results[1] = np.mean(deep_p_ar)
                                all_results[2] = np.mean(ada_br_ar)
                                all_results[3] = np.mean(ada_hr_ar)
                                all_results[4] = np.mean(var_RPM_ar)
                                all_results[5] = np.mean(var_HPM_ar)
                                all_results[6] = np.mean(rem_parameter_ar)
                                all_results[7] = np.mean(mov_dens_ar)
                                all_results[8] = np.mean(LF_ar)
                                all_results[9] = np.mean(HF_ar)
                                all_results[10] = np.mean(LFHF_ar)
                                all_results[11] = np.mean(sHF_ar)
                                all_results[12] = np.mean(sLFHF_ar)
                                all_results[13] = np.mean(tfRSA_ar)
                                all_results[14] = np.mean(tmHR_ar)
                                all_results[15] = np.mean(sfRSA_ar)
                                all_results[16] = np.mean(smHR_ar)
                                all_results[17] = np.mean(sdfRSA_ar)
                                all_results[18] = np.mean(sdmHR_ar)
                                all_results[19] = np.mean(stfRSA_ar)
                                all_results[20] = np.mean(stmHR_ar)
                                all_results[21] = np.mean(time_ar)
                                tmp_rest.append(all_results[23])
                                tmp_rest.append(all_results[22])
                                tmp_rest[2:24] = all_results[0:22]
                                print(tmp_rest)
                                print(len(tmp_rest))
                                sleep = rf2.predict(np.array(tmp_rest).reshape(1, -1))

                                # reset
                                tmp_rest = []
                                heart_ar = []
                                breath_ar = []
                                sleep_ar = []
                                bmi_ar = []
                                deep_p_ar = []
                                ada_br_ar = []
                                ada_hr_ar = []
                                var_RPM_ar = []
                                var_HPM_ar= []
                                rem_parameter_ar = []
                                mov_dens_ar = []
                                LF_ar = []
                                HF_ar = []
                                LFHF_ar = []
                                sHF_ar = []
                                sLFHF_ar = []
                                tfRSA_ar = []
                                tmHR_ar = []
                                sfRSA_ar = []
                                smHR_ar = []
                                sdfRSA_ar = []
                                sdmHR_ar = []
                                stfRSA_ar = []
                                stmHR_ar = []
                                time_ar = []
                                next_HM = False
                                recording_final(path_data, ct3[11:19], all_results, int(sleep[0]))
                            # # recording(path_data, vd, hr_rpm, br_rpm)
                            # if sec == "00":
                            #     recording_final(path_data, ct3[11:19], all_results)
                            
                        tmp_br = br_rpm
                        tmp_hr = hr_rpm
                        start_time = end_time

                except KeyboardInterrupt:
                    print("Interrupt")
            
            with open(path_range_bin, "a",newline="") as csvFile:
                writer = csv.writer(csvFile, dialect = "excel")
                writer.writerow([rangeBuf])