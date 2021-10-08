import serial
from mmWave import vitalsign

class globalV:
	count = 0
	hr = 0.0
	br = 0.0
	def __init__(self, count):
		self.count = count

port = serial.Serial("COM3",baudrate = 921600, timeout = 0.5)
gv = globalV(0)
vts = vitalsign.VitalSign(port)

def uartGetTLVdata():
	port.flushInput()
	while True:
		(dck , vd, rangeBuf) = vts.tlvRead(False)
		vs = vts.getHeader()
		if dck:
			print("unwrapPhasePeak_mm:{0:.4f}".format(vd.unwrapPhasePeak_mm))
uartGetTLVdata()
