# -*- coding: utf-8 -*-
from math import sqrt
from utils import *


def calEuclidDist(A, B):
    """
    :param A, B: two vectors
    :return: the Euclidean distance of A and B
    """
    return sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def dtw(M1, M2):
    """
    Compute Dynamic Time Warping(DTW) of two mfcc sequences.
    :param M1, M2: two mfcc sequences
    :return: the minimum distance and the wrap path
    """
    # length of two sequences
    M1_len = len(M1)
    M2_len = len(M2)
    cost_0 = np.zeros((M1_len + 1, M2_len + 1))
    cost_0[0, 1:] = np.inf
    cost_0[1:, 0] = np.inf
    # Initialize the array size to M1_len * M2_len
    cost = cost_0[1:, 1:]
    for i in range(M1_len):
        for j in range(M2_len):
            cost[i, j] = calEuclidDist(M1[i], M2[j])
    # dynamic programming to calculate cost matrix
    for i in range(M1_len):
        for j in range(M2_len):
            cost[i, j] += min([cost_0[i, j], \
                               cost_0[min(i + 1, M1_len - 1), j], \
                               cost_0[i, min(j + 1, M2_len - 1)]])

    # calculate the warp path
    if len(M1) == 1:
        path = np.zeros(len(M2)), range(len(M2))
    elif len(M2) == 1:
        path = range(len(M1)), np.zeros(len(M1))
    else:
        i, j = np.array(cost_0.shape) - 2
        path_1, path_2 = [i], [j]
        # path_1, path_2 with the minimum cost is what we want
        while (i > 0) or (j > 0):
            arg_min = np.argmin((cost_0[i, j], cost_0[i, j + 1], cost_0[i + 1, j]))
            if arg_min == 0:
                i -= 1
                j -= 1
            elif arg_min == 1:
                i -= 1
            else:
                j -= 1
            path_1.insert(0, i)
            path_2.insert(0, j)
        # convert to array
        path = np.array(path_1), np.array(path_2)
    # the minimum distance is the normalized distance
    return cost[-1, -1] / sum(cost.shape), path


def train_model_dtw(train_dir):
    """
    :param train_dir: directory of train audios
    :return: the trained mfcc model
    """
    model = []
    for digit in range(10):
        digit_dir = os.path.join(train_dir, "digit_" + str(digit))
        file_list = os.listdir(digit_dir)
        mfcc_list = []
        # read .wav files and calculate MFCC to mfcc_list
        for j in range(len(file_list)):
            file_path = os.path.join(digit_dir, file_list[j])
            if file_path[-4:] == ".wav":
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    wav_mfcc = mfcc(file_path)
                    mfcc_list.append(wav_mfcc)

        # set the first sequence as standard, merge all training sequences
        mfcc_count = np.zeros(len(mfcc_list[0]))
        mfcc_all = np.zeros(mfcc_list[0].shape)
        for i in range(len(mfcc_list)):
            # calculate the wrap path between standard and each template
            _, path = dtw(mfcc_list[0], mfcc_list[i])
            for j in range(len(path[0])):
                mfcc_count[int(path[0][j])] += 1
                mfcc_all[int(path[0][j])] += mfcc_list[i][path[1][j]]

        # Generalization by averaging the templates
        model_digit = np.zeros(mfcc_all.shape)
        for i in range(len(mfcc_count)):
            for j in range(len(mfcc_all[i])):
                model_digit[i][j] = mfcc_all[i][j] / mfcc_count[i]
        # return model with models of 0-9 digits
        model.append(model_digit)
    return model


def predict_dtw(model, file_path):
    """
    :param model: trained model
    :param file_path: path of .wav file
    :return: digit
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mfcc_feat = mfcc(file_path)
    # Iterate, find the digit with the minimum distance
    digit = 0
    min_dist, _ = dtw(model[0], mfcc_feat)
    for i in range(1, len(model)):
        dist, _ = dtw(model[i], mfcc_feat)
        if dist < min_dist:
            digit = i
            min_dist = dist
    return str(digit)


if __name__ == '__main__':
    train_dir = "./processed_train_records"
    test_dir = "./processed_test_records"
    model = train_model_dtw(train_dir)
    count = 0
    pred_true = 0
    for i in range(10):
        digit_dir = os.path.join(test_dir, "digit_" + str(i))
        file_list = os.listdir(digit_dir)
        for j in range(len(file_list)):
            file_path = os.path.join(digit_dir, file_list[j])
            if file_path[-4:] == ".wav":
                count += 1
                pred = predict_dtw(model, file_path)
                if file_path[-5] == pred:
                    pred_true += 1
    print("----------Dynamic Time Warping (DTW)----------")
    print("Train num: 160, Test num: %d, Predict true num: %d"%(count, pred_true))
    print("Accuracy: %.2f"%(pred_true / count))