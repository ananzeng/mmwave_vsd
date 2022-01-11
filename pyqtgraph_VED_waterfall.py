#=============================================
# File Name: pyqtgraph_VED_waterfall.py
#
# Requirement:
# Hardware: BM201-ISK or BM501-AOP
# Firmware: 
# config file: 
# lib: vehicleODR
# radar installation: wall momunt
#
# plot tools: pyqtgraph
# Plot Target (V8) in 3D figure 
# Vital Energy Detection
# type: Raw data
# Baud Rate: playback: 119200
#			 real time: 921600
#=============================================

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from mmWave import vehicleODR
#import vehicleODR

import serial
from threading import Thread

from datetime import date,datetime,time
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import os


class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
	def __init__(self, X, Y, Z, text):
		gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
		self.text = text
		self.X = X
		self.Y = Y
		self.Z = Z

	def setGLViewWidget(self, GLViewWidget):
		self.GLViewWidget = GLViewWidget

	def setText(self, text):
		self.text = text
		self.update()

	def setX(self, X):
		self.X = X
		self.update()

	def setY(self, Y):
		self.Y = Y
		self.update()

	def setZ(self, Z):
		self.Z = Z
		self.update()

	def paint(self):
		self.GLViewWidget.qglColor(QtCore.Qt.cyan)
		self.GLViewWidget.renderText(round(self.X), round(self.Y), round(self.Z), self.text)


st = datetime.now()
sim_startFN = 0
sim_stopFN = 0


##################### Parameter ################################### 
 

app = QtGui.QApplication([])

##################### init Water fall frame ##############################
traces = dict()
wf = gl.GLViewWidget()
wf.opts['distance'] = 40
wf.setWindowTitle('VED waterfall')
wf.setGeometry(0, 50, 500, 400)
wf.show()

'''
gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
#gx.translate(-10, 0, 0)
gx.translate(-10, 0, 10)
wf.addItem(gx)
'''

gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -48, 15)
gy.setSize(48, 36)
gy.setSpacing(1, 1)

wf.addItem(gy)

gz = gl.GLGridItem()
gz.translate(0, -18, 0)
gz.setSize(48, 66)
gz.setSpacing(1, 1)

wf.addItem(gz)

'''
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -48, 10)
gy.setSize(48, 24)
gy.setSpacing(1, 1)

wf.addItem(gy)

gz = gl.GLGridItem()
gz.translate(0, -18, 0)
gz.setSize(48, 66)
gz.setSpacing(1, 1)

wf.addItem(gz)
'''


######### for traces initial ##############################
n = 200
m = 200
y = np.linspace(-10, 10, n)
x = np.linspace(-10, 10, m)

phase = 0

yA = np.zeros(48)
xA = np.linspace(-24, 24, 48)

for i in range(n):
	yi = np.array([y[i]] * m)
	d = np.sqrt(x ** 2 + yi ** 2)
	z = 10 * np.cos(d + phase) / (d + 1)
	pts = np.vstack([x, yi, z]).transpose()
	traces[i] = gl.GLLinePlotItem(pos=pts, color=(0,0,0,0), width=(i + 1) / 10, antialias=True)
	wf.addItem(traces[i])

########## set data for 3d plot ####################

def set_plotdata(name, points, color, width):
	traces[name].setData(pos=points, color=color, width=width)

##################### water fall frame update ################################

def updateWF():
	global x,y,v8A
	if len(v8A) != 3072:
		return
	np.set_printoptions(precision=2)
	zA = np.array(v8A).reshape(64,48)
	#print(zA)
	for i in range(len(zA)):
		pts = np.vstack((xA,yA+ float(i) - 48.0,zA[i])).transpose()
		col = np.zeros((48,4))
		for j in range(48):
			#col[j] = colorMapping(j)
			col[j] = colorMapping(zA[i,j])
			#col[j] = colorMapping(yy[j])
		set_plotdata(name=i, points=pts,color= col,width= 3.0)


#
#****************** use USB-UART ************************************
#
#port = serial.Serial("/dev/ttyUSB0",baudrate = 921600, timeout = 0.5)
#
#for Jetson nano UART port
#port = serial.Serial("/dev/ttyTHS1",baudrate = 921600, timeout = 0.5) 
#
#for pi 4 UART port
#port = serial.Serial("/dev/ttyS0",baudrate = 921600, timeout = 0.5)
#
#Drone Object Detect Radar initial 
#port = serial.Serial("/dev/tty.usbmodemGY0052534",baudrate = 921600, timeout = 0.5)
#port = serial.Serial("/dev/tty.usbmodem14103",baudrate = 115200 , timeout = 0.5)  
#port = serial.Serial("/dev/tty.usbmodemGY0050674",baudrate = 921600, timeout = 0.5)  
#port = serial.Serial("/dev/tty.SLAB_USBtoUART3",baudrate = 921600, timeout = 0.5)  
port = serial.Serial("COM4",baudrate = 921600, timeout = 0.5)

#for NUC ubuntu 
#port = serial.Serial("/dev/ttyACM1",baudrate = 921600, timeout = 0.5)



radar = vehicleODR.VehicleODR(port)

v8len = 0
v9len = 0
v10len = 0

def colorMapping(inp):
	#print(inp)
	cx = (0,0,1,1)
	if inp < 0.2:
		cx = (0, 0, 0.7, 0.8) # blue
	elif inp >= 0.2 and inp < 0.6:
		cx = (0, 0, 1, 1)
	elif inp >= 0.6 and inp < 1.4:
		cx = (0,0.13,0.78,1)
	elif inp >= 1.4 and inp < 2.0:
		cx = (0,0.78,0,1)
	elif inp >= 2.0 and inp < 2.8:
		cx = (1,0.5,0,1)
	else: # R 20
		cx = (1,0,0,1)
	return cx


def jb_mapF32(inp, xMin, xMax, yMin, yMax):
	slope = (yMax - yMin) / (xMax - xMin)
	return (inp - xMin) * slope + yMin
	 
def jb_sigmod(inp, gain, threshold):
	x = gain * (inp - threshold)
	return inp * (1 / (1 + np.exp(-x)))

def jb_normalDistribute(inp, gain, mean, sigma):
	x = (inp - mean) / sigma
	return inp * np.exp(-x*x/2.0) * gain

def jb_limiter(inp, xMin, xMax, yMin, yMax):
	return jb_mapF32(inp, xMin, xMax, yMin, yMax)

v8A = []

def update():    
	global v8A
	if len(v8A) == 3072:
		print("======updateWF==================:{:}".format(len(v8A)))
		updateWF()


t = QtCore.QTimer()
t.timeout.connect(update)
t.start(150)
fn = 0 
prev_fn = 0

# ----------------------- 新增 ----------------------- 
save = 1  # 要儲存設 1 不存設 0
path = "./saved_img"
num = len(os.listdir(path))

def radarExec():
	global v9len,v10len,v8len,prev_fn,flag,uFlag,fn,v8A, num, path

	v8 = []
	v9 = []
	v10 = []
	flag = True
	(dck,v8,v9,v10) = radar.tlvRead(False)
	
	hdr = radar.getHeader()
	fn = hdr.frameNumber
	
	if  fn != prev_fn:
		print("--------------{:}-----------".format(fn))
		prev_fn = fn
		v8len = len(v8)
		v9len = len(v9)
		v10len = len(v10)
		
		if v8len == 3072:
			print("Sensor Data: [v8,v9,v10]:[{:d},{:d},{:d}]".format(v8len,v9len,v10len))
			
			#vs2 = np.log(v8) #np.sqrt(np.sqrt(v8))

			#print("===> Normalized={:.2f}, {:.2f}".format(max(v8), min(v8)))
		
			#v8A = (vs2 /  np.linalg.norm(vs2)) * 100.0
			vs = np.array(v8)
			vs = jb_mapF32(vs, 0.0, 32000.0/ 4.0, 0.0, 2.8)
            
			#vs = jb_sigmod(vs, 2.0, 32000.0 / 10000.0)
			jb_gain = 40.0 * 10000000
			jb_mean = 0.0
			vs = jb_normalDistribute(vs, jb_gain, jb_mean, 32000.0 / 2000.0)
			jb_limitValue = 2.5 
			vs = jb_limiter(vs, 0, jb_gain, 0, jb_limitValue) 
			
			
			v8A = vs

			# ----------------------- 新增 -----------------------
			cv2img = np.zeros((64,48))
			count = 0
			for i in range(64):
				for j in range(48):
					cv2img[i][j] = v8A[count]
					count += 1
 
			heatmapshow = None
			heatmapshow = cv2.normalize(cv2img, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
			heatmapshow1 = heatmapshow.copy()
			heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
			heatmapshow = cv2.cvtColor(heatmapshow, cv2.COLOR_BGR2GRAY)
			#kernel = np.ones((3,3), np.uint8)
			#heatmapshow = cv2.dilate(heatmapshow, kernel, iterations = 1)
			heatmapshow = cv2.resize(heatmapshow, (1024, 768), interpolation=cv2.INTER_AREA)
			heatmapshow = cv2.flip(heatmapshow, 1)

			# print(np.shape(heatmapshow))
			# _, heatmapshow = cv2.threshold(heatmapshow, 127, 255, cv2.THRESH_BINARY)
			
			# contours, hierarchy = cv2.findContours(heatmapshow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# if len(contours) == 1:
			# 	order = 0
			# else:
			# 	area_temp = []
			# 	for i in range(len(contours)):
			# 		area1 = cv2.contourArea(contours[i])
			# 		area_temp.append(area1)
			# 	order = np.argmax(np.array(area_temp))
			# rect = cv2.minAreaRect(contours[order])
			# box = cv2.boxPoints(rect)
			# box = np.int0(box)
			# print(box)

			# file_mask_copy1 = np.expand_dims(file_mask_copy, axis=2)
			# file_mask_copy2 = np.concatenate((file_mask_copy1, file_mask_copy1, file_mask_copy1), axis=-1)

			heatmapshow = np.expand_dims(heatmapshow, axis=2)
			heatmapshow = np.concatenate((heatmapshow, heatmapshow, heatmapshow), axis=-1)
			cv2.imshow("Heatmap", heatmapshow)

			#cv2.rectangle(heatmapshow, (box[1]), (box[3]), (0, 0, 255), 2)
			print(np.min(heatmapshow))
			cv2.imshow("Heatmap", heatmapshow)
			cv2.waitKey(1)
			cv2.imwrite(path + '/heatmap_' + str(num) + '.png', heatmapshow1)
			num += 1
			# if save == 1:
			# 	if not os.path.isdir(path):
			# 		os.mkdir(path)
					
			# 	cv2.imwrite(path + '/heatmap_' + str(num) + '.jpg', heatmapshow)
			# 	num += 1
			# 	cv2.waitKey(0)
			# else:
			# 	cv2.waitKey(0)
			# --------------------------------------------------

		
	port.flushInput()

		 
def uartThread(name):
	port.flushInput()
	while True:
		radarExec()
					
thread1 = Thread(target = uartThread, args =("UART",))
thread1.setDaemon(True)
thread1.start()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore,'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
