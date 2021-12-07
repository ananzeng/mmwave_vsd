import serial
import datetime,time
import numpy as np
import vitalsign_v2
import csv
import os
import pickle #pickle模組
from  combine_svm import detect_Breath, substitute

class globalV:
	count = 0
	hr = 0.0
	br = 0.0
	def __init__(self, count):
		self.count = count
#---------------------------------read txt----------------------------------
def read_txt(path):
    data=np.loadtxt(path)
    data=data.astype(np.float32)
    return data

if __name__ == "__main__":
    next_YMD = False
    coco = True  # 時間開關
    switch = True
    time.sleep(4)
    print("111111111111111111111111111111111111111111111")
    time.sleep(1)
    # Data initial
    # port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5) # Data Port
    # #initial global value
    # gv = globalV(0)

    """ 左臻 """
    # # UART Write initial
    # portw = serial.Serial("COM5",baudrate = 115200, timeout = 0.5) # User UART :notebook:COM6,computer:COM4
    # Read_Radar_parameter = open("./Radar_parameter.txt", "r")

    # for parameter in iter(Read_Radar_parameter):
    #     portw.write((parameter.encode(encoding="utf-8")).strip(b'\n').replace(b"\\n",b"\n").replace(b"\\r",b"\r"))
    #     time.sleep(1)
    #     print((parameter.encode(encoding="utf-8")).strip(b'\n').replace(b"\\n",b"\n").replace(b"\\r",b"\r"))

    # Read_Radar_parameter.close()
    # # Data initial
    # port = serial.Serial("COM6",baudrate = 921600, timeout = 0.5) # Data Port
    # #initial global value
    # gv = globalV(0)
    """ ------------------------------------ """

    """ 久邦 """
    # Data initial
    port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5) # Data Port
    #initial global value
    gv = globalV(0)
    """ ------------------------------------ """

    vts = vitalsign_v2.VitalSign(port)
    # --------------------------------------------------
    folder='./dataset_sleep'									#資料庫名稱
    name = '1124test'
    distance = str(0.8)									#資料要放進哪個距離的資料夾
    people =  folder +'/'+ name 
    if not os.path.isdir(people):
        os.mkdir(people)
    people2 =  folder +'/'+ name +'/'+ distance
    if not os.path.isdir(people2):
        os.mkdir(people2)

    raw_sig = []  # 訊號的窗格
    energe_br = []  # 呼吸能量的窗格
    energe_hr = []  # 心跳能量的窗格
    heart_ti = []  #TI心跳
    breath_ti = []  #TI呼吸
    tmp_br = 0  # 初始化前一秒呼吸律
    tmp_hr = 0  # 初始化前一秒心律
    port.flushInput() #丟棄接收緩存中的所有數據

    #--------------------write csv-----------------------
    data_number = str(int(len(os.listdir( folder +'/'+ name +'/'+ distance +'/'))/2)) # 錄製的哪一筆資料
    path_data = folder +'/'+ name +'/'+ distance +'/'+ data_number +".csv"
    path_range_bin = folder +'/'+ name +'/'+ distance +'/Range_bins_'+ data_number +".csv"
    with open(path_data, 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(['rangeBinIndexMax','rangeBinIndexPhase','maxVal','processingCyclesOut','processingCyclesOut1',
                            'rangeBinStartIndex','rangeBinEndIndex','unwrapPhasePeak_mm','outputFilterBreathOut','outputFilterHeartOut',
                            'heartRateEst_FFT','heartRateEst_FFT_4Hz','heartRateEst_xCorr','heartRateEst_peakCount','breathingRateEst_FFT',
                            'breathingEst_xCorr','breathingEst_peakCount','confidenceMetricBreathOut','confidenceMetricBreathOut_xCorr','confidenceMetricHeartOut',
                            'confidenceMetricHeartOut_4Hz','confidenceMetricHeartOut_xCorr','sumEnergyBreathWfm','sumEnergyHeartWfm','motionDetectedFlag',
                            'rsv[0]','rsv[1]','rsv[2]','rsv[3]','rsv[4]','rsv[5]','rsv[6]','rsv[7]','rsv[8]','rsv[9]','datetime', "heart", "breath"]) # 寫入csv一列資料
    with open(path_range_bin, 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(['rangeBinInde'])
    count = 0    
    
    print("Recoding data...")
    time_Start = time.time()

    # 每秒算一次
    ct = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # 時間格式為字串
    start_year = ct[0:4]  
    start_month = ct[5:7]
    start_day = ct[8:10]
    start_time = int(ct[11:13])*3600 + int(ct[14:16])*60 + int(ct[17:19])

    while True:
        (dck , vd, rangeBuf) = vts.tlv_read(False)  # 是否顯示[Message TLV header]
        vs = vts.getHeader()
        if dck:
            # ct = datetime.datetime.now().strftime('%H:%M:%S') # 時間格式為字串
            # # ct = datetime_dt.strftime("%H:%M:%S")  # 格式化日期
            # start_time = int(ct[0:2])*3600 + int(ct[3:5])*60 + int(ct[6:8])

            # ct = datetime.datetime.now()
            raw_sig.append(vd.unwrapPhasePeak_mm)
            # energe_br.append(vd.sumEnergyBreathWfm)
            # energe_hr.append(vd.sumEnergyHeartWfm)
            heartRateEst_FFT_mean = np.mean(vd.heartRateEst_FFT)
            heartRateEst_xCorr_mean = np.mean(vd.heartRateEst_xCorr)
            breathingEst_FFT_mean = np.mean(vd.breathingRateEst_FFT)
            breathingEst_xCorr_mean = np.mean(vd.breathingEst_xCorr)
            heart_ti.append(vd.rsv[1])
            breath_ti.append(vd.rsv[0])
            a = [[1.5, 0.125, 0.55, 20, 5, 2, 22, 17], [1.5, 0.9, 1.9, 20, 9, 2, 5, 4]]
            a = np.array(a)
            hr_rate = 0
            br_rate = 0
            time_End = time.time()
            # 計時 40 s
            if coco:
                print(f"Elapsed time (sec): {round(time_End - time_Start, 3)}")
                
            if len(raw_sig) <= 40*20:
                with open(path_data, 'a',newline='') as csvFile:
                    writer = csv.writer(csvFile, dialect = "excel")
                    writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                                    vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                                    vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                                    vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                                    vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                                   vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct[11:19], 0 ,0])
                                   
            elif len(raw_sig) > 40*20:
                coco = False
                try:
                    current_window_sig = raw_sig[-40*20:]
                    raw_sig.pop(0)
                    current_heart_ti = heart_ti[-40*20:]
                    heart_ti.pop(0)
                    current_breath_ti = breath_ti[-40*20:]
                    breath_ti.pop(0)
                    # -------- 廢除 -------- 
                    # current_window_ebr = energe_br[-3*20:]
                    # current_window_ehr = energe_hr[-3*20:]

                    # 秒數
                    ct2 = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # 時間格式為字串
                    end_year = ct2[0:4]  
                    end_month = ct2[5:7]
                    end_day = ct2[8:10]
                    end_time = int(ct2[11:13])*3600 + int(ct2[14:16])*60 + int(ct2[17:19])
                    # time_End = time.time()

                    # 當隔天為: 過年、過月、過天
                    if int(end_year) - int(start_year) >= 1:
                        # 換起始時間
                        start_year = end_year
                        start_month = end_month
                        start_day = end_day
                        start_time = end_time
                        next_YMD == True
                    if int(end_month) - int(start_month) >= 1:
                        # 換起始時間
                        start_month = end_month
                        start_day = end_day
                        start_time = end_time
                        next_YMD == True
                    if int(end_day) - int(start_day) >= 1:
                        # 換起始時間
                        start_day = end_day
                        start_time = end_time
                        next_YMD == True

                    if (end_time - start_time >= 1) or next_YMD == True:
                        next_YMD == False
                        start_time = end_time

                        # 呼吸
                        br_rate, index_of_fftmax = detect_Breath(current_window_sig, a[0][:])
                        with open('save/svm_br_office_all.pickle', 'rb') as f:
                            clf = pickle.load(f)
                            #print("br_svm_predict ", clf.predict(([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])))
                            svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                            if svm_predict == 1:
                                br_rate = int(np.mean(current_breath_ti))

                        # 心跳
                        hr_rate ,index_of_fftmax = detect_Breath(current_window_sig, a[1][:])
                        with open('save/svm_hr_office_all.pickle', 'rb') as f:
                            clf = pickle.load(f)
                            #print("hr_svm_predict ", clf.predict(([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])))
                            svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                            if svm_predict == 1:
                                hr_rate = int(np.mean(current_heart_ti))

                        br_rpm = np.round(br_rate)
                        hr_rpm = np.round(hr_rate)
                        br_rpm = substitute(tmp_br, br_rpm, 1)
                        hr_rpm = substitute(tmp_hr, hr_rpm, 0)

                        # print("TIME：", ct2)
                        print(f"Breathe Rate per minute: {br_rpm}")
                        print(f"Heart Rate per minute: {hr_rpm}")
                        print()
                        ct3 = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # 時間格式為字串
                        with open(path_data, 'a',newline='') as csvFile:
                            writer = csv.writer(csvFile, dialect = "excel")
                            writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                                                vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                                                vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                                                vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                                                vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                                                vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct3[11:19], hr_rpm, br_rpm])
                        tmp_br = br_rpm
                        tmp_hr = hr_rpm
                    else:
                        ct3 = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # 時間格式為字串
                        with open(path_data, 'a',newline='') as csvFile:
                            writer = csv.writer(csvFile, dialect = "excel")
                            writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                                            vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                                            vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                                            vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                                            vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                                            vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct3[11:19], 0 ,0])
                        '''
                        # 判別有沒有呼吸與有沒有人 (憋氣可以判別)
                        if np.mean(current_window_ebr) > 50000000 and np.mean(current_window_ehr) > 50:
                            br_rate, index_of_fftmax = detect_Breath(current_window_sig, a[0][:])
                            with open('save/svm_br_office_all.pickle', 'rb') as f:
                                clf = pickle.load(f)
                                #print("br_svm_predict ", clf.predict(([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])))
                                svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                                if svm_predict == 1:
                                    br_rate = int(np.mean(current_breath_ti))

                            hr_rate ,index_of_fftmax = detect_Breath(current_window_sig, a[1][:])
                            with open('save/svm_hr_office_all.pickle', 'rb') as f:
                                clf = pickle.load(f)
                                #print("hr_svm_predict ", clf.predict(([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])))
                                svm_predict = clf.predict([[index_of_fftmax,heartRateEst_FFT_mean,heartRateEst_xCorr_mean]])
                                if svm_predict == 1:
                                    hr_rate = int(np.mean(current_heart_ti))

                            br_rpm = np.round(br_rate)
                            hr_rpm = np.round(hr_rate)
                            print("TIME：", ct)
                            print(f"Breathe Rate per minute: {br_rpm}")
                            print(f"Heart Rate per minute: {hr_rpm}")
                            print()
                            with open(path_data, 'a',newline='') as csvFile:
                                writer = csv.writer(csvFile, dialect = "excel")
                                writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                                                    vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                                                    vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                                                    vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                                                    vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                                                    vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct, hr_rpm, br_rpm]) 					
                        elif np.mean(current_window_ebr) < 50000000 and np.mean(current_window_ehr) < 50:
                            print("No People")
                            hr_rate = 0
                            br_rate = 0
                            '''
                        
                # Ctrl C 中斷
                except KeyboardInterrupt:
                    print("Interrupt")
                                    
            with open(path_range_bin, 'a',newline='') as csvFile:
                writer = csv.writer(csvFile, dialect = "excel")
                writer.writerow([rangeBuf])