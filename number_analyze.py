from losscal import calculate_l1_loss
import matplotlib.pyplot as plt
import numpy as np
def heart_analyze(gt, pr):
    heart_array = []
    array_50_60 = []
    array_gt_50_60 = []
    array_60_70 = []
    array_gt_60_70 = []
    array_70_80 = []
    array_gt_70_80 = []
    array_80_90 = []
    array_gt_80_90 = []
    array_90_100 = []
    array_gt_90_100 = []

    for index,i in enumerate(pr):
        if int(i)>=50 and int(i)<60:
            array_50_60.append(i)
            array_gt_50_60.append(gt[index])
        if int(i)>=60 and int(i)<70:
            array_60_70.append(i)
            array_gt_60_70.append(gt[index])
        if int(i)>=70 and int(i)<80:
            array_70_80.append(i)
            array_gt_70_80.append(gt[index])
        if int(i)>=80 and int(i)<90:
            array_80_90.append(i)
            array_gt_80_90.append(gt[index])
        if int(i)>=90 and int(i)<100:
            array_90_100.append(i)
            array_gt_90_100.append(gt[index])

    heart_array.append(calculate_l1_loss(array_gt_50_60, array_50_60))
    heart_array.append(calculate_l1_loss(array_gt_60_70, array_60_70))
    heart_array.append(calculate_l1_loss(array_gt_70_80, array_70_80))
    heart_array.append(calculate_l1_loss(array_gt_80_90, array_80_90))
    heart_array.append(calculate_l1_loss(array_gt_90_100, array_90_100))

    print("L1 Loss of 50~60", heart_array[0])
    print("L1 Loss of 60~70", heart_array[1])
    print("L1 Loss of 70~80", heart_array[2])
    print("L1 Loss of 80~90", heart_array[3])
    print("L1 Loss of 90~100", heart_array[4])
    return heart_array


def breath_analyze(gt, pr):
    breath_array = []
    array_12 = [i for i in pr if int(i) == 12]
    array_gt_12 = [gt[index] for index, i in enumerate(pr) if int(i) == 12]
    array_13 = [i for i in pr if int(i) == 13]
    array_gt_13 = [gt[index] for index, i in enumerate(pr) if int(i) == 13]
    array_14 = [i for i in pr if int(i) == 14]
    array_gt_14 = [gt[index] for index, i in enumerate(pr) if int(i) == 14]
    array_15 = [i for i in pr if int(i) == 15]
    array_gt_15 = [gt[index] for index, i in enumerate(pr) if int(i) == 15]
    array_16 = [i for i in pr if int(i) == 16]
    array_gt_16 = [gt[index] for index, i in enumerate(pr) if int(i) == 16]
    array_17 = [i for i in pr if int(i) == 17]
    array_gt_17 = [gt[index] for index, i in enumerate(pr) if int(i) == 17]
    array_18 = [i for i in pr if int(i) == 18]
    array_gt_18 = [gt[index] for index, i in enumerate(pr) if int(i) == 18]
    array_19 = [i for i in pr if int(i) == 19]
    array_gt_19 = [gt[index] for index, i in enumerate(pr) if int(i) == 19]
    array_20 = [i for i in pr if int(i) == 20]
    array_gt_20 = [gt[index] for index, i in enumerate(pr) if int(i) == 20]
    array_21 = [i for i in pr if int(i) == 21]
    array_gt_21 = [gt[index] for index, i in enumerate(pr) if int(i) == 21]
    array_22 = [i for i in pr if int(i) == 22]
    array_gt_22 = [gt[index] for index, i in enumerate(pr) if int(i) == 22]

    breath_array.append(round(calculate_l1_loss(array_12, array_gt_12), 1))
    breath_array.append(round(calculate_l1_loss(array_13, array_gt_13), 1))
    breath_array.append(round(calculate_l1_loss(array_14, array_gt_14), 1))
    breath_array.append(round(calculate_l1_loss(array_15, array_gt_15), 1))
    breath_array.append(round(calculate_l1_loss(array_16, array_gt_16), 1))
    breath_array.append(round(calculate_l1_loss(array_17, array_gt_17), 1))
    breath_array.append(round(calculate_l1_loss(array_18, array_gt_18), 1))
    breath_array.append(round(calculate_l1_loss(array_19, array_gt_19), 1))
    breath_array.append(round(calculate_l1_loss(array_20, array_gt_20), 1))
    breath_array.append(round(calculate_l1_loss(array_21, array_gt_21), 1))
    breath_array.append(round(calculate_l1_loss(array_22, array_gt_22), 1))

    print("L1 Loss of 12", breath_array[0])
    print("L1 Loss of 13", breath_array[1])
    print("L1 Loss of 14", breath_array[2])
    print("L1 Loss of 15", breath_array[3])
    print("L1 Loss of 16", breath_array[4])
    print("L1 Loss of 17", breath_array[5])
    print("L1 Loss of 18", breath_array[6])
    print("L1 Loss of 19", breath_array[7])
    print("L1 Loss of 20", breath_array[8])
    print("L1 Loss of 21", breath_array[9])
    print("L1 Loss of 22", breath_array[10])
    return breath_array