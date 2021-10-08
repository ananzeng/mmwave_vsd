gt = [84, 76, 83, 88, 86, 86, 91, 73, 90, 84, 88, 77, 74, 74, 74, 72, 73, 74, 78, 77]
Butterworth_bandpass_filter = [75, 79, 83, 84, 81, 84, 87, 72, 88, 61, 68, 80, 80, 78, 73, 64, 74, 75, 70, 73]
IIR_Bessel_bandpass_filter = [77, 77, 83, 84, 82, 84, 87, 76, 88, 68, 74, 80, 74, 75, 70, 65, 77, 78, 71, 77]
FIR_highpass_filter = [75, 75, 83, 84, 81, 84, 86, 63, 86, 77, 72, 75, 68, 76, 72, 71, 73, 74, 61, 70]

IIR_butter_bandpass_filter = [75, 79, 84, 84, 81, 84, 87, 72, 88, 61, 68, 80, 80, 78, 73, 64, 74, 75, 70, 73]
IIR_cheby1_bandpass_filter = [72, 77, 83, 81, 81, 84, 86, 75, 86, 65, 70, 75, 77, 75, 75, 70, 74, 75, 75, 71]
IIR_cheby2_bandpass_filter = [80, 77, 83, 84, 82, 84, 86, 77, 89, 81, 81, 78, 76, 74, 73, 72, 74, 78, 79, 79]


temp = 0
for i in range(len(gt)):
    temp += abs(gt[i] - IIR_cheby2_bandpass_filter[i])
print("l1 loss", temp/len(gt))

def calculate_l1_loss(gt, pr):
    temp = 0
    for i in range(len(gt)):
        temp += abs(gt[i] - pr[i])
    return temp/len(gt)
'''
import numpy as np
import matplotlib.pyplot as plt
students = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
math_scores = [84, 76, 83, 88, 86, 86, 91, 73, 90, 84, 88, 77, 74, 74, 74, 72, 73, 74, 78, 77]
history_scores = [75, 79, 83, 84, 81, 84, 87, 72, 88, 61, 68, 80, 80, 78, 73, 64, 74, 75, 70, 73]
x = np.arange(len(students))
width = 0.3
plt.bar(x, history_scores, width, label='Ground Truth', edgecolor = 'white')
plt.bar(x + width, math_scores, width, color='mediumseagreen', label='Predict', edgecolor = 'white')
plt.xticks(x + width / 2, students)
plt.ylabel('Heart')
plt.xlabel('Samples')
#plt.title('L1 Loss')
plt.legend(loc='upper left')
plt.grid(True)
plt.grid(color='white',    
         linestyle='-',
         linewidth=1,
         alpha=0.3) 

plt.show()

temp1 = []
for i in range(len(gt)):
    temp1.append(gt[i] - pr2[i])


plt.plot(np.arange(len(gt)), np.array(temp1), '-o')     # red dotted line (no marker)
plt.axhline(y=10, xmin=0, xmax=1,color='red')
plt.axhline(y=-10, xmin=0, xmax=1, color='red')
plt.xticks(np.linspace(0,19,20)) 
plt.xlim([0,19])
plt.ylim([-20,20])
plt.title('L1 Loss')
plt.ylabel('Heart Rate')
plt.xlabel('Samples')
plt.show()
'''