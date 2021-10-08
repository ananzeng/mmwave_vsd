# vitalsign
# ver:0.0.10
# 2020/02/04
#
#
import serial
#import time
import struct
#import numpy as np

class vsos:
    rangeBinIndexMax  = 0 
    rangeBinIndexPhase = 0 
    maxVal  = float(0.0)
    processingCyclesOut =  0 
    processingCyclesOut1 =  0 
    rangeBinStartIndex  =  0 
    rangeBinEndIndex    =  0  
    unwrapPhasePeak_mm  = float(0.0)
    outputFilterBreathOut = float(0.0)
    outputFilterHeartOut = float(0.0)
    heartRateEst_FFT     = float(0.0)
    heartRateEst_FFT_4Hz  = float(0.0)
    heartRateEst_xCorr   = float(0.0)
    heartRateEst_peakCount  = float(0.0)
    breathingRateEst_FFT   = float(0.0)
    breathingEst_xCorr     = float(0.0)
    breathingEst_peakCount  = float(0.0)
    confidenceMetricBreathOut  = float(0.0)
    confidenceMetricBreathOut_xCorr  = float(0.0)
    confidenceMetricHeartOut   = float(0.0)
    confidenceMetricHeartOut_4Hz  = float(0.0)
    confidenceMetricHeartOut_xCorr  = float(0.0)
    sumEnergyBreathWfm = float(0.0)
    sumEnergyHeartWfm  = float(0.0)
    motionDetectedFlag = float(0.0)
    rsv  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
	
class header:
    version = ""
    totalPackLen =0
    tvlHeaderLen = 8
    platform = ""
    frameNumber = 0
    timeCpuCycles = 0
    numDetectedObj = 0
    numTLVs = 0
    rsv = 0
	
	
class VitalSign:
    magicWord =  [b'\x02',b'\x01',b'\x04',b'\x03',b'\x06',b'\x05',b'\x08',b'\x07',b'\0x99']
    port = ""

    hdr = header
    vs = vsos()
    '''
    version = ""
    totalPackLen =0
    tvlHeaderLen = 8
    platform = ""
    frameNumber = 0
    timeCpuCycles = 0
    numDetectedObj = 0
    numTLVs = 0
    '''
        
    def __init__(self,port):
        self.port = port
        print("***vital sign init***")
        
    def vital_port(self):
        print("------vital Sign ---- ok:" + self.port)
        
        
    def showHeader(self):
        print("**********************") 
        print("PID:\t%d "%(self.hdr.frameNumber))
        print("Version:\t%x "%(self.hdr.version))
        print("TLV:\t\t%d "%(self.hdr.numTLVs))
        print("Detect Obj:\t%d "%(self.hdr.numDetectedObj))
        print("Platform:\t%X "%(self.hdr.platform))
        print("TotalPackLen:\t%d "%(self.hdr.totalPackLen))

    def getvitalSignsOutputStats(self):
        return self.vs
        
    def getHeader(self):
        return self.hdr

    def vitalSignsOutputStats(self,buf):
        
        try:
            (self.vs.rangeBinIndexMax,
            self.vs.rangeBinIndexPhase, 
            self.vs.maxVal,
            self.vs.processingCyclesOut,
            self.vs.processingCyclesOut1,
            self.vs.rangeBinStartIndex,
            self.vs.rangeBinEndIndex,
            self.vs.unwrapPhasePeak_mm,
            self.vs.outputFilterBreathOut,
            self.vs.outputFilterHeartOut,
            self.vs.heartRateEst_FFT,
            self.vs.heartRateEst_FFT_4Hz,
            self.vs.heartRateEst_xCorr,
            self.vs.heartRateEst_peakCount,
            self.vs.breathingRateEst_FFT,
            self.vs.breathingEst_xCorr,
            self.vs.breathingEst_peakCount,
            self.vs.confidenceMetricBreathOut,
            self.vs.confidenceMetricBreathOut_xCorr,
            self.vs.confidenceMetricHeartOut,
            self.vs.confidenceMetricHeartOut_4Hz,
            self.vs.confidenceMetricHeartOut_xCorr,
            self.vs.sumEnergyBreathWfm,
            self.vs.sumEnergyHeartWfm,
            self.vs.motionDetectedFlag,
            self.vs.rsv[0],self.vs.rsv[1],self.vs.rsv[2],
            self.vs.rsv[3],self.vs.rsv[4],self.vs.rsv[5],
            self.vs.rsv[6],self.vs.rsv[7],self.vs.rsv[8],
            self.vs.rsv[9]) = struct.unpack('2Hf2H2H28f', buf)
        except:
            print("vsos error found:")
            return (False,self.vs)

        return (True,self.vs)
        
        

                    
        
    def tlvRead(self,disp):
        #print("---tlvRead---")
        vsdata = vsos
        idx = 0
        lstate = 'idle'
        sbuf = b""
        
        #*******Message TLV Header 1 *******
        mmWaveType1 = 0
        length1 = 0
        #********Message TLV Header 2******
        mmWaveType2 = 0
        length2 = 0

        while True:
            ch = self.port.read()
            if lstate == 'idle':
                if ch == self.magicWord[idx]:
                    #print("*** magicWord:"+ "{:02x}".format(ord(ch)) + ":" + str(idx))
                    #GPIO.output(21, True)	
                    idx += 1
                    if idx == 8:			
                        idx = 0
                        lstate = 'header'
                        rangeProfile = b""
                        sbuf = b""		
                        #print("--vital header")
                else:
                    idx = 0
                    rangeProfile = b""#出錯的地方
                    return (False,vsdata, rangeProfile)
        
            elif lstate == 'header':
                sbuf += ch
                idx += 1
                if idx == 40:	
                    #print("------header-----")
                    #print(":".join("{:02x}".format(c) for c in sbuf))
                    #print("------header end -----")  
                    # [header - Magicword] + [Message TLV header] 
                    try:   		  
                        '''(self.version , self.totalPackLen, self.platform ,
                        self.frameNumber, self.timeCpuCycles, self.numDetectedObj,
                        self.numTLVs, rsv, mmWaveType1, length1) = struct.unpack('10I', sbuf)
                        '''
                        
                        (self.hdr.version , self.hdr.totalPackLen, self.hdr.platform ,
                        self.hdr.frameNumber, self.hdr.timeCpuCycles, self.hdr.numDetectedObj, 
                        self.hdr.numTLVs, self.hdr.rsv, mmWaveType1, length1) = struct.unpack('10I', sbuf) 
        
                    
                    except:
                        print("Improper TLV structure found: ")
                        return (False,vsdata, rangeProfile)
                        
                    if disp == True:
                        self.showHeader()
                            #-----[Message TLV header] = 8 bytes ----
                        print("mmWave Type 1:\t%d "%(mmWaveType1))
                        print("Data 1 Len:\t%d "%(length1)) 
                    
                    sbuf = b""
                    idx = 0
                    lstate = 'vsos'
                    #if length1 > 288: 
                    length1 = 128 
                    #print("header=>data1=>length:"%(length1))
                        
                elif idx > 40:
                    idx = 0
                    lstate = 'idle'
                    return (False, vsdata ,rangeProfile)
                    
                                
            elif lstate == 'vsos': #128 Vital Signs Output Status
                sbuf += ch
                idx += 1
                if idx == length1: #128
                    #print("-----vital Signs Output Stats----") 
                    #print(":".join("{:02x}".format(c) for c in sbuf))
                    vflag, vsdata = self.vitalSignsOutputStats(sbuf)
                    #print("-----vital Signs Output Stats end ---")
                    if not vflag:
                        lstate = 'idle'
                        return (False,vsdata, rangeProfile)

                    idx = 0 
                    lstate = 'mTLVh'
                    sbuf = b""
                
                elif idx > length1:
                    idx = 0
                    lstate = 'idle'
                    return (False,vsdata, rangeProfile)
                            
            elif lstate == 'mTLVh':
                sbuf += ch
                idx += 1
                if idx == 8:
                    #print(":".join("{:02x}".format(c) for c in sbuf))
                    idx = 0  
                    try:   		  
                        mmWaveType2 , length2 = struct.unpack('2I',sbuf) 
                    except:
                        print("Improper Type2, length structure found: ")
                        return (False,vsdata, rangeProfile)
                    if disp == True:
                        print("mmWave Type 2:\t%d "%(mmWaveType2))
                        print("Data 2 Len:\t%d "%(length2))
                    #print("mTLVh---state")  
                    if length2 > 252:
                        length2 = 252
                    lstate = 'rangeProfile'
                    rangeProfile = b""
                    #[Message TLV header] = 8 bytes
                    
                    
                elif idx > 8:
                    idx = 0
                    lstste = 'idle'
                    
            elif lstate == 'rangeProfile':
                rangeProfile += ch
                idx += 1
                if idx == length2:
                    idx = 0
                    try:
                        fmt = '{:d}h'.format(int(length2/2))
                        xd = struct.unpack(fmt,rangeProfile)
                    except:
                        print("Improper rangeProfile data found: ")
                        return (False,vsdata, rangeProfile)
                    if disp == True:
                        print("---------rangeProfile:" + str(len(rangeProfile)))
                        print(":".join("{:02x}".format(c) for c in rangeProfile))
                    #GPIO.output(21, False)
                    #print(xd)
                    return (True,vsdata, list(xd))
                elif idx > length2:
                    idx = 0
                    lstate = 'idle'


