from losscal import calculate_l1_loss

def heart_analyze(gt, pr):
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


    print("L1 Loss of 50~60", calculate_l1_loss(array_gt_50_60, array_50_60))
    print("L1 Loss of 60~70", calculate_l1_loss(array_gt_60_70, array_60_70))
    print("L1 Loss of 70~80", calculate_l1_loss(array_gt_70_80, array_70_80))
    print("L1 Loss of 80~90", calculate_l1_loss(array_gt_80_90, array_80_90))
    print("L1 Loss of 90~100", calculate_l1_loss(array_gt_90_100, array_90_100))
