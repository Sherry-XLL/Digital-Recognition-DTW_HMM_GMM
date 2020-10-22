# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import librosa


def mfcc(wav_path, delta = 2):
    """
    Read .wav files and calculate MFCC
    :param wav_path: path of audio file
    :param delta: derivative order, default order is 2
    :return: mfcc
    """
    y, sr = librosa.load(wav_path)
    # MEL frequency cepstrum coefficient
    mfcc_feat = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)
    ans = [mfcc_feat]
    # Calculate the 1st derivative
    if delta >= 1:
        mfcc_delta1 = librosa.feature.delta(mfcc_feat, order = 1, mode ='nearest')
        ans.append(mfcc_delta1)
    # Calculate the 2nd derivative
    if delta >= 2:
        mfcc_delta2 = librosa.feature.delta(mfcc_feat, order = 2, mode ='nearest')
        ans.append(mfcc_delta2)
    return np.transpose(np.concatenate(ans, axis = 0),[1,0])