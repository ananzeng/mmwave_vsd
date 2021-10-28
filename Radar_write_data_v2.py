import serial
import datetime,time
import numpy as np
import vitalsign_v2
import csv
import os
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
    time.sleep(5)
    print("111111111111111111111111111111111111111111111")
    time.sleep(1)
    # Data initial
    port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5) # Data Port
    print(port)
    #initial global value
    gv = globalV(0)
    vts = vitalsign_v2.VitalSign(port)
    # --------------------------------------------------
    folder='./dataset'									#資料庫名稱
    name = '1022test'
    distance = str(0.8)												#資料要放進哪個距離的資料夾
    people =  folder +'/'+ name 
    if not os.path.isdir(people):
        os.mkdir(people)
    people2 =  folder +'/'+ name +'/'+ distance
    people2_gt_hr =  folder +'/'+ name +'/'+ "gt_hr"
    people2_gt_br =  folder +'/'+ name +'/'+ "gt_br"
    if not os.path.isdir(people2):
        os.mkdir(people2)
    if not os.path.isdir(people2_gt_hr):
        os.mkdir(people2_gt_hr)
    if not os.path.isdir(people2_gt_br):
        os.mkdir(people2_gt_br)
    # --------------------------------------------------
    time_Start = time.time()
    time_End = 0
    print ("Start : %s" % time.ctime())

    # UART : 50 ms , uart Get TLV data : VitalSign
    pt = datetime.datetime.now()
    ct = datetime.datetime.now()
    port.flushInput() #丟棄接收緩存中的所有數據
    #--------------------write csv-----------------------
    data_number = str(int(len(os.listdir( folder +'/'+ name +'/'+ distance +'/'))/2)) #錄製的哪一筆資料
    path_data = folder +'/'+ name +'/'+ distance +'/'+ data_number +".csv"
    path_data_txt_hr = folder +'/'+ name +'/'+ "gt_hr" +'/'+ data_number +".txt"
    path_data_txt_br = folder +'/'+ name +'/'+ "gt_br" +'/'+ data_number +".txt"
    path_range_bin = folder +'/'+ name +'/'+ distance +'/Range_bins_'+ data_number +".csv"
    with open(path_data, 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(['rangeBinIndexMax','rangeBinIndexPhase','maxVal','processingCyclesOut','processingCyclesOut1',
                            'rangeBinStartIndex','rangeBinEndIndex','unwrapPhasePeak_mm','outputFilterBreathOut','outputFilterHeartOut',
                            'heartRateEst_FFT','heartRateEst_FFT_4Hz','heartRateEst_xCorr','heartRateEst_peakCount','breathingRateEst_FFT',
                            'breathingEst_xCorr','breathingEst_peakCount','confidenceMetricBreathOut','confidenceMetricBreathOut_xCorr','confidenceMetricHeartOut',
                            'confidenceMetricHeartOut_4Hz','confidenceMetricHeartOut_xCorr','sumEnergyBreathWfm','sumEnergyHeartWfm','motionDetectedFlag',
                            'rsv[0]','rsv[1]','rsv[2]','rsv[3]','rsv[4]','rsv[5]','rsv[6]','rsv[7]','rsv[8]','rsv[9]','datetime']) # 寫入csv一列資料
    with open(path_range_bin, 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow(['rangeBinInde'])
    
    while(int(time_End - time_Start) != 400*0.1):
        pt = datetime.datetime.now()
        (dck , vd, rangeBuf) = vts.tlv_read(False)  #是否顯示[Message TLV header]
        #print(dck)
        vs = vts.getHeader()
        if dck:
            ct = datetime.datetime.now()
            gv.br = vd.breathingRateEst_FFT
            gv.hr = vd.heartRateEst_FFT
            gv.RM = vd.rangeBinIndexMax
            gv.RP = vd.rangeBinIndexPhase
            with open(path_data, 'a',newline='') as csvFile:
                writer = csv.writer(csvFile, dialect = "excel")
                writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                                    vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                                    vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                                    vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                                    vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                                    vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct])
            time_End = time.time()
            with open(path_range_bin, 'a',newline='') as csvFile:
                writer = csv.writer(csvFile, dialect = "excel")
                writer.writerow([rangeBuf])
        
        if(int(time_End - time_Start)%1 == 0):
            print('time:',int(time_End - time_Start))

    print("done")
    #檢測rangeBinIndexPhase是否平穩 超過3算誤判
    temp_array = []
    with open(path_data, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            temp_array.append(row[1])
        temp_array = temp_array[1:] 
    temp_array = np.array(temp_array).astype('float32')
    print("STD：", np.std(temp_array)) 
    print('Maximum error: ', np.max(temp_array) - np.min(temp_array))
    print("done!")

    # 判別錯誤資料
    input_gt_hr = int(input('輸入你的心律Ground Truth：'))
    input_gt_br = int(input('輸入你的呼吸律Ground Truth：'))
    if input_gt_hr == 0 and input_gt_br == 0:
        os.remove(path_data)
        os.remove(path_range_bin)
        print('Data cleaning completed!')
    else:
        # Heart
        f = open(path_data_txt_hr, 'w')
        f.write(str(input_gt_hr))
        f.close()
        # Breath
        f = open(path_data_txt_br, 'w')
        f.write(str(input_gt_br))
        f.close()
    print("現在有", str(len(os.listdir(people2_gt_br))), "資料")
