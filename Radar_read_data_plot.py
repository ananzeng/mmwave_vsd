import serial
import struct
import datetime,time
import numpy as np
# from mmWave import vitalsign
import vitalsign_v2
import json
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.fftpack import fft, fftfreq, fftshift,ifft
from scipy.signal import butter, lfilter,iirfilter

SampleRate = 50

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
'''
        plot_ans(rangeProfile_cplx_T = rangeProfile_cplx_T[i,:], \
                phaseindex = phaseindex, \
                fig = fig_angle.add_subplot(round((len(rangeProfile_cplx_T))/3+0.5),3,i+1), \
                fig2 = fig_unwarp.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig3 = fig_FFT.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig4 = fig_range.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig5 = fig_unwarp2.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                # fig6 = fig_iq.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1, projection='polar'), \
                fig7 = fig_unwarp3.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                index = str(round((index_m + float(int(i/3)/10)),1)) + 'm')
'''
def plot_ans(rangeProfile_cplx_T,phaseindex,fig,fig2,fig3,fig4,fig5,fig7,index):
    #------------------------------- angle
    ax = fig
    ant1_Ph  = np.angle(rangeProfile_cplx_T) #计算复数的辐角主值。1+1j->45*pi/180
    ax.plot(np.arange(len(ant1_Ph)),ant1_Ph,label="$sin(x)$",linewidth=1)
    ax.set_title('Range to angle index = '+ index,fontsize=10)
    ax.set_ylabel('Radians',fontsize=8)
    ax.set_xlabel('Frame Number',fontsize=8)
    #------------------------------- unwrap phase
    ax2 = fig2
    ant1_UnWrapPh_no_index = np.unwrap(ant1_Ph,discont=np.pi) #, discont=2 * np.pi
    ax2.plot(np.arange(len(ant1_UnWrapPh_no_index)),ant1_UnWrapPh_no_index,label="$sin(x)$",linewidth=1)
    ax2.set_title('Range to unwrap Phase index = '+ index,fontsize=10)
    ax2.set_ylabel('Radians',fontsize=8)
    ax2.set_xlabel('Frame Number',fontsize=8)
    #------------------------------- FFT
    ax3 = fig3
    # ant1_Ph_BPF = butter_bandpass_filter(ant1_Ph, 0.8, 4.0, SampleRate, order=5)
    # angleFFT = fft(ant1_Ph_BPF)
    ant1_UnWrapPh_BPF = butter_bandpass_filter(ant1_UnWrapPh_no_index, 0.8, 4.0, SampleRate, order=5)
    # angleFFT = fft(ant1_UnWrapPh_BPF)
    # N = len(angleFFT)
    # T = 1/SampleRate
    # xf = np.linspace(0.0, 1.0/(T*2), N//2) # '//'= 整數除法
    # ax3.plot(xf, 2.0/N * np.abs(angleFFT[0:N//2]))
    ax3.set_title('FFT Magnitude = '+ index ,fontsize=10)
    ax3.set_ylabel('Amplitude[dB]',fontsize=8)
    ax3.set_xlabel('Frequency [Hz]',fontsize=8)
    # number of signal points
    N = len(ant1_UnWrapPh_BPF)
    # sample spacing
    T = 1.0 / SampleRate
    yf = fft(ant1_UnWrapPh_BPF)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)
    ax3.plot(xf, 1.0/N * np.abs(yplot))
    ax3.grid()
    #------------------------------- Magnitude of the Range Profiles
    ax4 = fig4
    ax4.plot(np.arange(len(rangeProfile_cplx_T)),abs(rangeProfile_cplx_T),label="$sin(x)$",linewidth=1)
    ax4.set_title('Range to Magnitude index = '+ index,fontsize=10)
    ax4.set_ylabel('Radians',fontsize=8)
    ax4.set_xlabel('Frame Number',fontsize=8)
    #------------------------------- unwrap phase
    ax5 = fig5
    ant1_index = ant1_Ph - phaseindex
    #print("ant1_Ph123", ant1_Ph)
    #print("phaseindex", phaseindex)
    #print("ant1_index", ant1_index)
    ant1_UnWrapPh_Yes_index  = np.unwrap(ant1_index,discont=np.pi) #, discont=2 * np.pi
    ax5.plot(np.arange(len(ant1_UnWrapPh_Yes_index)),ant1_UnWrapPh_Yes_index,label="$sin(x)$",linewidth=1)
    ax5.set_title('Range to unwrap Phase index = '+ index,fontsize=10)
    ax5.set_ylabel('Radians',fontsize=8)
    ax5.set_xlabel('Frame Number',fontsize=8)
    #------------------------------- IQ
    # ax = fig6
    # A = abs(rangeProfile_cplx_T)
    # R = np.angle(rangeProfile_cplx_T)*A
    # ax.plot(R, A)
    # # ax.set_rmax(2)
    # # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    # ax.grid(True)
    # ax.set_title("A line plot on a polar axis", va='bottom')
    #-------------------------------- umwarp phase difference
    ax7 = fig7
    ant1_diff = []
    for i in range(0,len(ant1_index) - 1):
        ant1_diff.append(ant1_index[i + 1] - ant1_index[i])
    ax7.plot(np.arange(len(ant1_diff)),ant1_diff,label="$sin(x)$",linewidth=1)
    ax7.set_title('Range to angle index = '+ index,fontsize=10)
    ax7.set_ylabel('Radians',fontsize=8)
    ax7.set_xlabel('Frame Number',fontsize=8)

if __name__ == "__main__":
    folder='./dataset'
    name = 'zheng-liang'
    distance = str(0.8)
    # data_number = str(int(len(os.listdir( folder +'/'+ name +'/'+ distance +'/'))/2) - 1) #錄製的哪一筆資料
    data_number = str(0)
    path_data =  folder +'/'+ name +'/'+ distance +'/'+ data_number +".csv"
    path_range_bin =  folder +'/'+ name +'/'+ distance +'/Range_bins_'+ data_number +".csv"

    #----------------------------------plot answer------------------------------------
    dataset = pd.read_csv(path_data) #(21,35)(rows,cols)(value,name)
    #-------------------------------- rangeBuf Convert -------------------------------
    data_range_bin = pd.read_csv(path_range_bin).values
    #print("data_range_bin", data_range_bin)
    rangeProfile_cplx = []
    for i in range(0,len(data_range_bin)):
        rangeBuf_list = []
        data_range_bin_rows =  ("".join("".join(("".join(str(pd.read_csv(path_range_bin).values[i]).split("'")).split("["))).split("]")).split(','))
        for j in range(0,len(data_range_bin_rows),2):
            rangeBuf_list.append(complex(int(data_range_bin_rows[0 + j]),int(data_range_bin_rows[1 + j])))
        rangeProfile_cplx.append(rangeBuf_list)
    #print("rangeProfile_cplx", rangeProfile_cplx) #生成復數
    #------------------------------------- 2D ----------------------------------------
    #------------------------------- outputFilterHeartOut phase
    unwrapPhasePeak_mm_x =np.arange(len(dataset['unwrapPhasePeak_mm'].values))
    # plt.plot(x,data[:],"r-o",label="$sin(x)$",linewidth=1)
    plt.subplot(5,2,1)
    plt.title('unwrapPhasePeak_mm')
    plt.plot(unwrapPhasePeak_mm_x,dataset['unwrapPhasePeak_mm'].values,label="$sin(x)$",linewidth=1)
    plt.xlabel('samples(1 second = 20)')
    plt.ylabel("Amplitude")
    #------------------------------- outputFilterHeartOut phase
    heart_x =np.arange(len(dataset['outputFilterHeartOut'].values))
    # plt.plot(x,data[:],"r-o",label="$sin(x)$",linewidth=1)
    plt.subplot(5,2,2)
    plt.title('Heart')
    plt.plot(heart_x,dataset['outputFilterHeartOut'].values,label="$sin(x)$",linewidth=1)
    plt.xlabel('samples(1 second = 20)')
    plt.ylabel("Radians")
    #------------------------------- outputFilterBreathOut phase
    breath_x =np.arange(len(dataset['outputFilterBreathOut'].values))
    plt.subplot(5,2,3)
    plt.title('Breath')
    plt.plot(breath_x,dataset['outputFilterBreathOut'].values,label="$sin(x)$",linewidth=1)
    plt.xlabel('samples(1 second = 20)')
    plt.ylabel("Radians")
    #------------------------------- Magnitude of the Range Profiles
    plt.subplot(5,2,4)
    #print("np.abs(rangeProfile_cplx)", np.abs(rangeProfile_cplx))
    #print()
    plt.pcolormesh(np.arange(len(rangeProfile_cplx[0])), np.arange(len(rangeProfile_cplx)), np.abs(rangeProfile_cplx))
    plt.colorbar()
    plt.title('Magnitude of the Range Profiles',fontsize=12)
    plt.ylabel('Frame Number',fontsize=10)
    plt.xlabel('Range Bins',fontsize=10)
    #------------------------------- Selecteds column of the Range Profile
    plt.subplot(5,2,5)
    for i in range (0,len(rangeProfile_cplx)):
        plt.plot(np.arange(len(rangeProfile_cplx[i])),np.abs(rangeProfile_cplx[i]),label="$sin(x)$",linewidth=1)
    plt.title('Selecteds column of the Range Profile',fontsize=12)
    plt.xlabel('Range Bins',fontsize=10)
    plt.ylabel("Radians",fontsize=10)
    plt.subplots_adjust(wspace =0.5, hspace =0.5)
    #-------------------------------range-bin phase index
    plt.subplot(5,2,6)
    phaseindex_x =np.arange(len(dataset['rangeBinIndexPhase'].values))
    plt.plot(phaseindex_x,dataset['rangeBinIndexPhase'].values,label="$sin(x)$",linewidth=1)
    plt.title('Range-bin phase index',fontsize=12)
    plt.ylabel('Radians',fontsize=10)
    plt.xlabel('samples(1 second = 20)',fontsize=10)
    #------------------------------- breath FFT
    plt.subplot(5,2,7)
    # breathFFT = fft(dataset['outputFilterBreathOut'].values)
    # N = len(dataset['outputFilterBreathOut'].values)
    # T = 1/20
    # xf = np.linspace(0.0, 1.0/(T*2), N//2) # '//'= 整數除法
    # plt.plot(xf, 2.0/N * np.abs(breathFFT[0:N//2]))
    plt.title('breathFFT Magnitude',fontsize=12)
    plt.ylabel('Amplitude[dB]',fontsize=10)
    plt.xlabel('Frequency [Hz]',fontsize=10)

    # number of signal points
    N = len(dataset['outputFilterBreathOut'].values)
    # sample spacing
    T = 1.0 / SampleRate
    yf = fft(dataset['outputFilterBreathOut'].values)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)
    plt.plot(xf, 1.0/N * np.abs(yplot))
    plt.grid()

    #------------------------------- heart FFT
    plt.subplot(5,2,8)
    # heartFFT = fft(dataset['outputFilterHeartOut'].values)
    # N = len(dataset['outputFilterHeartOut'].values)
    # T = 1/20
    # xf = np.linspace(0.0, 1.0/(T*2), N//2) # '//'= 整數除法
    # plt.plot(xf, 2.0/N * np.abs(heartFFT[0:N//2]))
    plt.title('heartFFT Magnitude',fontsize=12)
    plt.ylabel('Amplitude[dB]',fontsize=10)
    plt.xlabel('Frequency [Hz]',fontsize=10)

    # number of signal points
    N = len(dataset['outputFilterHeartOut'].values)
    # sample spacing
    T = 1.0 / SampleRate
    yf = fft(dataset['outputFilterHeartOut'].values)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)
    plt.plot(xf, 1.0/N * np.abs(yplot))
    plt.grid()
    plt.subplots_adjust(wspace =0.8, hspace =0.8)
    #------------------------------- Range-bins unwrap Phase
    rangeProfile_cplx_T = np.array(rangeProfile_cplx).T
    fig_angle = plt.figure()
    fig_unwarp = plt.figure()
    fig_FFT = plt.figure()
    fig_range = plt.figure()
    fig_unwarp2 = plt.figure()
    # fig_iq = plt.figure()
    fig_unwarp3 = plt.figure()
    fig_angle.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_angle.suptitle('angle', fontsize=16)
    fig_unwarp.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_unwarp.suptitle('unwrap phase', fontsize=16)
    fig_FFT.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_FFT.suptitle('FFT', fontsize=16)
    fig_range.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_range.suptitle('range', fontsize=16)
    fig_unwarp2.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_unwarp2.suptitle('unwrap index - phase', fontsize=16)
    # fig_iq.subplots_adjust(wspace =0.8, hspace =0.8)
    # fig_iq.suptitle('IQ', fontsize=16)
    fig_unwarp3.subplots_adjust(wspace =0.8, hspace =0.8)
    fig_unwarp3.suptitle('unwrap phase difference', fontsize=16)

    index_num = int((int(dataset['rangeBinStartIndex'].values[0]) - 2)/3)
    if(index_num == 0 ):
        index_m = 0.1
    elif(index_num > 0 ):
        index_m = 0.1 + round(float(index_num/10),1)
    phaseindex = []
    for i in range(0,len(dataset['rangeBinIndexPhase'].values)):
        phaseindex.append((np.pi/180)*dataset['rangeBinIndexPhase'].values[i]) 
    #print(len(rangeProfile_cplx_T))
    #print(len(rangeProfile_cplx_T[0]))

    #print(len(rangeProfile_cplx))
    #print(len(rangeProfile_cplx[0]))
    for i in range(0,len(rangeProfile_cplx_T)):
        plot_ans(rangeProfile_cplx_T = rangeProfile_cplx_T[i,:], \
                phaseindex = phaseindex, \
                fig = fig_angle.add_subplot(round((len(rangeProfile_cplx_T))/3+0.5),3,i+1), \
                fig2 = fig_unwarp.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig3 = fig_FFT.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig4 = fig_range.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                fig5 = fig_unwarp2.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                # fig6 = fig_iq.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1, projection='polar'), \
                fig7 = fig_unwarp3.add_subplot(round(len(rangeProfile_cplx_T)/3+0.5),3,i+1), \
                index = str(round((index_m + float(int(i/3)/10)),1)) + 'm')
    
  
    #------------------------------------ 3D ----------------------------------------
    #------------------------------- RangeIndex
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # rangeProfile_cplx_T = np.array(rangeProfile_cplx).T
    # x_3D = np.arange(len(rangeProfile_cplx_T[0]))
    # y_3D = np.arange(len(rangeProfile_cplx_T))
    # X_3D,Y_3D = np.meshgrid(x_3D,y_3D)

    # ant1_Ph  = np.angle(rangeProfile_cplx_T)
    # ant1_UnWrapPh  = np.unwrap(ant1_Ph, discont=np.pi) 
    # ax.plot_wireframe(X_3D, Y_3D, ant1_UnWrapPh, rstride=1, cstride=1000, lw=.5)
    # # surf = ax.plot_surface(X_3D, Y_3D, ant1_UnWrapPh, antialiased=True, cmap=cm.coolwarm, rstride=1, cstride=1000, shade=False, lw=.5)
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title('rangeIndex = all')
    # ax.set_zlabel('Radians') #,fontsize=16
    # ax.set_xlabel('samples(1 second = 20)')
    # ax.set_ylabel('range-bins(0.1m = 3)')

    #------------------------------- Magnitude of the Range Profiles
    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    x_3D = np.arange(len(rangeProfile_cplx[0]))
    y_3D = np.arange(len(rangeProfile_cplx))
    X_3D,Y_3D = np.meshgrid(x_3D,y_3D)
    Z_3D = np.abs(rangeProfile_cplx)
    ax.plot_wireframe(X_3D, Y_3D, Z_3D, rstride=1, cstride=1000, lw=.5)
    # ax.xaxis.set_major_formatter(FuncFormatter(to_percent))#修改座標刻度
    
    # x_3D = np.arange(len(rangeProfile_cplx[0]))
    # y_3D = np.arange(len(rangeProfile_cplx))
    # X_3D,Y_3D = np.meshgrid(x_3D,y_3D)
    # Z_3D = np.abs(rangeProfile_cplx)
    # surf = ax.plot_surface(X_3D, Y_3D, Z_3D, antialiased=True, cmap=cm.coolwarm, rstride=1, cstride=1000, shade=False, lw=.5)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.axhline
    ax.set_title('Magnitude of the Range Profiles')
    ax.set_zlabel('Radians') #,fontsize=16
    ax.set_xlabel('range-bins(0.1m = 3)')
    ax.set_ylabel('samples(1 second = 20)')

    plt.show()
