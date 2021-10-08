import numpy as np
def get_arg():
    arg_array = []
    filter_RemoveImpulseNoise = np.arange(1.2, 2, 0.2)
    iir_bandpass_filter_1_arg1 = np.arange(0.1, 0.2, 0.05)
    iir_bandpass_filter_1_arg2 = np.arange(0.5, 0.7, 0.05)
    iir_bandpass_filter_1_arg3 = np.arange(3, 5, 1)
    iir_bandpass_filter_1_arg4 = ["butter", "ellip", "cheby1", "cheby2"]
    MLR = np.arange(1, 12, 1)
    feature_compress = np.arange(1, 25, 1)
    candidate_search = np.arange(1, 25, 1)
    '''
    for i in filter_RemoveImpulseNoise:
        for j in iir_bandpass_filter_1_arg1:
            for k in iir_bandpass_filter_1_arg2:
                for l in iir_bandpass_filter_1_arg3:
                    for m in iir_bandpass_filter_1_arg4:
                        for n in MLR:
                            for o in feature_compress:
                                for p in candidate_search:
                                    temp = []
                                    temp.append(i)
                                    temp.append(j)
                                    temp.append(k)
                                    temp.append(l)
                                    temp.append(m)
                                    temp.append(n)
                                    temp.append(o)
                                    temp.append(p)
                                    arg_array.append(temp)
    '''
    for i in filter_RemoveImpulseNoise:
        for j in iir_bandpass_filter_1_arg1:
            for k in iir_bandpass_filter_1_arg2:
                for l in iir_bandpass_filter_1_arg3:
                    for m in iir_bandpass_filter_1_arg4:
                                    temp = []
                                    temp.append(round(i, 2))
                                    temp.append(round(j, 2))
                                    temp.append(round(k, 2))
                                    temp.append(l)
                                    temp.append(m)
                                    arg_array.append(temp)
    print(arg_array)
    print(len(arg_array))
    print(len(arg_array[0]))
    return arg_array