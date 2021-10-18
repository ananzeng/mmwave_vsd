# gt = [84, 76, 83, 88, 86, 86, 91, 73, 90, 84, 88, 77, 74, 74, 74, 72, 73, 74, 78, 77]
# Butterworth_bandpass_filter = [75, 79, 83, 84, 81, 84, 87, 72, 88, 61, 68, 80, 80, 78, 73, 64, 74, 75, 70, 73]
# IIR_Bessel_bandpass_filter = [77, 77, 83, 84, 82, 84, 87, 76, 88, 68, 74, 80, 74, 75, 70, 65, 77, 78, 71, 77]
# FIR_highpass_filter = [75, 75, 83, 84, 81, 84, 86, 63, 86, 77, 72, 75, 68, 76, 72, 71, 73, 74, 61, 70]

# IIR_butter_bandpass_filter = [75, 79, 84, 84, 81, 84, 87, 72, 88, 61, 68, 80, 80, 78, 73, 64, 74, 75, 70, 73]
# IIR_cheby1_bandpass_filter = [72, 77, 83, 81, 81, 84, 86, 75, 86, 65, 70, 75, 77, 75, 75, 70, 74, 75, 75, 71]
# IIR_cheby2_bandpass_filter = [80, 77, 83, 84, 82, 84, 86, 77, 89, 81, 81, 78, 76, 74, 73, 72, 74, 78, 79, 79]

import numpy as np
import matplotlib.pyplot as plt

# temp = 0
# for i in range(len(gt)):
#     temp += abs(gt[i] - IIR_cheby2_bandpass_filter[i])
# print("l1 loss", temp/len(gt))

def calculate_l1_loss(gt, pr):
    temp = 0
    for i in range(len(gt)):
        temp += abs(gt[i] - pr[i])
    return temp/len(gt)

def range_hr(arr):
    ans = np.zeros(10)
    for i in arr:
        if i>=55 and i<60:
            ans[0] += 1
        elif i>=60 and i<65:
            ans[1] += 1
        elif i>=65 and i<70:
            ans[2] += 1
        elif i>=70 and i<75:
            ans[3] += 1
        elif i>=75 and i<80:
            ans[4] += 1
        elif i>=80 and i<85:
            ans[5] += 1
        elif i>=85 and i<90:
            ans[6] += 1
        elif i>=90 and i<95:
            ans[7] += 1
        elif i>=95 and i<100:
            ans[8] += 1
        elif i>=100 and i<105:
            ans[9] += 1
        
    return ans

def range_br(arr):
    ans = np.zeros(11)
    for i in arr:
        if i>=12 and i < 13:
            ans[0] += 1
        elif i==13 and i < 14:
            ans[1] += 1
        elif i==14 and i < 15:
            ans[2] += 1
        elif i==15 and i < 16:
            ans[3] += 1
        elif i==16 and i < 17:
            ans[4] += 1
        elif i==17 and i < 18:
            ans[5] += 1
        elif i==18 and i < 19:
            ans[6] += 1
        elif i==19 and i < 20:
            ans[7] += 1
        elif i==20 and i < 21:
            ans[8] += 1
        elif i==21 and i < 22:
            ans[9] += 1
        elif i>=22:
            ans[10] += 1
        
    return ans

# 畫LOSS
def diagram(pr, ti, current_type):
    if current_type == 'h':
        candidates=['50~60', '60~70', '70~80', '80~90', '90~100', '100~110']
        dia_title = "Heart rate"
    elif current_type == 'b':
        candidates=['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
        dia_title = "Breath rate"

    plt.figure()
    plt.xticks(range(len(candidates)), candidates)
    plt.plot(np.arange(len(ti)), ti, 'o-')
    plt.plot(np.arange(len(pr)), pr, 'o-')
    plt.title("Compare loss")
    plt.ylabel("Loss")
    plt.xlabel(dia_title)
    plt.legend(['TI', 'our'], loc='upper left')

def data_distribution(data_pr, data_ti, data_gt, current_type):
    if current_type == 'h':
        pr_array = range_hr(data_pr)
        ti_array = range_hr(data_ti)
        gt_array = range_hr(data_gt)
        candidates=['55~60', '60~65', '65~70', '70~75', '75~80', '80~85', '85~90', '90~95', '95~100', '100~105']
    elif current_type == 'b':
        pr_array = range_br(data_pr)
        ti_array = range_br(data_ti)
        gt_array = range_br(data_gt)
        candidates=['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']

    # 算法預測
    plt.figure()
    plt.xticks(range(len(candidates)), candidates)
    plt.bar(np.arange(len(pr_array)), pr_array, width=0.25, color='lightcoral')
    plt.title("our")
    plt.ylabel("Number")
    plt.xlabel("Heart rate")

    # TI預測
    plt.figure()
    plt.xticks(range(len(candidates)), candidates)
    plt.bar(np.arange(len(ti_array)), ti_array, width=0.25, color='cornflowerblue')
    plt.title("TI")
    plt.ylabel("Number")
    plt.xlabel("Heart rate")

    # Ground Truth
    plt.figure()
    plt.xticks(range(len(candidates)), candidates)
    plt.bar(np.arange(len(gt_array)), gt_array, width=0.25, color='g')
    plt.title("Ground Truth")
    plt.ylabel("Number")
    plt.xlabel("Heart rate")

    # 合在一起
    plt.figure()
    plt.xticks(range(len(candidates)), candidates)
    plt.bar(np.arange(len(pr_array)), pr_array, width=0.25, color='lightcoral')
    plt.bar(np.arange(len(ti_array))+0.25, ti_array, width=0.25, color='cornflowerblue')
    plt.bar(np.arange(len(gt_array))+0.5, gt_array, width=0.25, color='g')
    plt.legend(['our', 'TI', 'Ground Truth'], loc='upper left')
    plt.title("Compare")
    plt.ylabel("Number")
    plt.xlabel("Heart rate")

    plt.show()
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