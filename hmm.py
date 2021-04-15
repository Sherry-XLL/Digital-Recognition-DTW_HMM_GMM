# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


def mfcc(wav_path, delta=2):
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


def get_mfcc_data(train_dir):
    """
    :param train_dir: training directory
    :return: mfcc_data
    """
    mfcc_data = []
    for i in range(10):
        digit_mfcc = np.array([])
        digit = str(i)
        digit_dir = os.path.join(train_dir, 'digit_' + digit)
        train_files = [x for x in os.listdir(digit_dir) if x.endswith('.wav')]
        for file_name in train_files:
            file_path = os.path.join(digit_dir, file_name)
            # get mfcc feature and ignore the warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(file_path)
            # append mfcc to X
            if len(digit_mfcc) == 0:
                digit_mfcc = features_mfcc
            else:
                digit_mfcc = np.append(digit_mfcc, features_mfcc, axis=0)
        mfcc_data.append(digit_mfcc)

    return mfcc_data


def log_gaussian_prob(x, means, var):
    """
    Estimate the log Gaussian probability
    :param x: input array
    :param means: means of each state.
    :param var: variance of each state.
    :return: the log Gaussian probability
    """
    return (-0.5 * np.log(var) - np.divide(np.square(x - means), 2 * var) - 0.5 * np.log(2 * np.pi)).sum()


class HMM():
    """Hidden Markov Model.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    mfcc_data : array, shape (, 39)
        Mfcc data of training wavs.

    Attributes
    ----------
    init_prob : array, shape (n_components, )
        Initial probability over states.

    means : array, shape (n_components, 1)
        Mean parameters for each mixture component in each state.

    var : array, shape (n_components, 1)
        Variance parameters for each mixture component in each state.

    trans_mat : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    """
    def __init__(self, mfcc_data, n_components):
        # Initialization
        self.mfcc_data = mfcc_data
        self.init_prob = np.hstack((np.ones(1), np.zeros(n_components - 1)))
        self.n_components = n_components
        self.means = np.tile(np.mean(self.mfcc_data, axis=0), (n_components, 1))
        self.var = np.tile(np.var(self.mfcc_data, axis=0), (n_components, 1))
        self.states = []
        self.trans_mat = np.zeros((self.n_components, self.n_components))

    def viterbi(self, data):
        # Viterbi algorithm is used to solve decoding problem of HMM
        # That is to find the most possible labeled sequence of the observation sequence
        for index, single_wav in enumerate(data):
            n = single_wav.shape[0]
            labeled_seq = np.zeros(n, dtype=int)
            log_delta = np.zeros((n, self.n_components))
            psi = np.zeros((n, self.n_components))
            log_delta[0] = np.log(self.init_prob)
            for i in range(self.n_components):
                log_delta[0, i] += log_gaussian_prob(single_wav[0], self.means[i], self.var[i])
            log_trans_mat = np.log(self.trans_mat)
            for t in range(1, n):
                for j in range(self.n_components):
                    temp = np.zeros(self.n_components)
                    for i in range(self.n_components):
                        temp[i] = log_delta[t - 1, i] + log_trans_mat[i, j] + log_gaussian_prob(single_wav[t], self.means[j], self.var[j])
                    log_delta[t, j] = np.max(temp)
                    psi[t, j] = np.argmax(log_delta[t - 1] + log_trans_mat[:, j])
            labeled_seq[n - 1] = np.argmax(log_delta[n - 1])
            for i in reversed(range(n - 1)):
                labeled_seq[i] = psi[i + 1, labeled_seq[i + 1]]
            self.states[index] = labeled_seq

    def forward(self, data):
        # Computes forward log-probabilities of all states at all time steps.
        n = data.shape[0]
        m = self.means.shape[0]
        alpha = np.zeros((n, m))
        alpha[0] = np.log(self.init_prob) + np.array([log_gaussian_prob(data[0], self.means[j], self.var[j]) for j in range(m)])
        for i in range(1, n):
            for j in range(m):
                alpha[i, j] = log_gaussian_prob(data[i], self.means[j], self.var[j]) + np.max(
                    np.log(self.trans_mat[:, j].T) + alpha[i - 1])
        return alpha

    def forward_backward(self, data):
        # forward backward algorithm to estimate parameters
        gamma_0 = np.zeros(self.n_components)
        gamma_1 = np.zeros((self.n_components, data[0].shape[1]))
        gamma_2 = np.zeros((self.n_components, data[0].shape[1]))
        for index, single_wav in enumerate(data):
            n = single_wav.shape[0]
            labeled_seq = self.states[index]
            gamma = np.zeros((n, self.n_components))
            for t, j in enumerate(labeled_seq[:-1]):
                self.trans_mat[j, labeled_seq[t + 1]] += 1
                gamma[t, j] = 1
            gamma[n - 1, self.n_components - 1] = 1
            gamma_0 += np.sum(gamma, axis=0)
            for t in range(n):
                gamma_1[labeled_seq[t]] += single_wav[t]
                gamma_2[labeled_seq[t]] += np.square(single_wav[t])
        gamma_0 = np.expand_dims(gamma_0, axis=1)
        self.means = gamma_1 / gamma_0
        self.var = (gamma_2 - np.multiply(gamma_0, self.means ** 2)) / gamma_0
        for j in range(self.n_components):
            self.trans_mat[j] /= np.sum(self.trans_mat[j])

    def train(self, data, iter):
        # Function for training single Gaussian modal
        if iter == 0:
            for single_wav in data:
                n = single_wav.shape[0]
                init_seq = np.array([self.n_components * i / n for i in range(n)], dtype=int)
                self.states.append(init_seq)
        self.forward_backward(data)
        self.viterbi(data)

    def log_prob(self, data):
        # return log likelihood of single modal Gaussian
        n = data.shape[0]
        alpha = self.forward(data)
        return np.max(alpha[n - 1])


def train_model_hmm(train_dir, n_components, max_iter = 100):
    """
    :param train_dir: training directory
    :param n_components: The number of mixture components
    :param max_iter: The number of EM iterations to perform
    :return: hmm_models
    """
    hmm_models = []
    # get mfcc data
    mfcc_data = get_mfcc_data(train_dir)
    mfcc_dict = {}
    for i in range(10):
        digit = str(i)
        hmm_models.append(HMM(mfcc_data[i], n_components=n_components))
        digit_dir = os.path.join(train_dir, 'digit_' + digit)
        data = []
        train_files = [x for x in os.listdir(digit_dir) if x.endswith('.wav')]
        for file_name in train_files:
            file_path = os.path.join(digit_dir, file_name)
            # get mfcc feature and ignore the warning
            features_mfcc = mfcc(file_path)
            # append mfcc to data
            if len(data) == 0:
                data = [features_mfcc]
            else:
                data.append(features_mfcc)
        mfcc_dict[digit] = data
    # start iteration
    iter = 0
    lower_bound = -np.infty
    while iter <= max_iter:
        prev_lower_bound = lower_bound
        total_log_like = 0.0
        for i in range(10):
            mfcc_i = mfcc_dict[str(i)]
            hmm_models[i].train(mfcc_i, iter)
            for single_wav in mfcc_i:
                total_log_like += hmm_models[i].log_prob(single_wav)
        lower_bound = total_log_like
        # The iteration stops when total_log_like converges
        if abs(lower_bound - prev_lower_bound) < 0.1:
            break
        print("Iteration:", iter, " log_prob:", lower_bound)
        iter += 1

    return hmm_models


def predict_hmm(hmm_models, test_dir):
    """
    :param hmm_models: trained hmm models
    :param test_dir: testing directory
    """
    count = 0
    pred_true = 0
    confusion_matrix = np.zeros((10,10)).astype(int)
    # append all test records and start digit recognition
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            # Make sure the suffix is correct and avoid the influence of hidden files
            if os.path.splitext(file)[1] == '.wav':
                test_files.append(os.path.join(root, file))
    for test_file in test_files:
        # get mfcc feature and ignore the warning
        features_mfcc = mfcc(test_file)
        # calculate the score and get the maximum score
        sum_prob = []
        for i in range(10):
            prob_i = hmm_models[i].log_prob(features_mfcc)
            sum_prob.append(prob_i)
        pred = str(np.argmax(np.array(sum_prob)))
        ground_truth = os.path.splitext(test_file)[0][-1]
        print("pred: ", pred, "   ground_truth: ", ground_truth)
        if pred == ground_truth:
            pred_true += 1
        confusion_matrix[int(ground_truth), int(pred)] += 1
        count += 1

    print("---------- HMM (Hidden Markov Model) ----------")
    print("confusion_matrix: \n", confusion_matrix)
    print("Train num: %d, Test num: %d, Predict true num: %d" % (200-count, count, pred_true))
    print("Accuracy: %.2f" % (pred_true * 100 / count))


def predict_gmm(gmm_models, test_dir):
    """
    :param gmm_models: trained gmm models
    :param test_dir: testing directory
    """
    count = 0
    pred_true = 0
    confusion_matrix = np.zeros((10,10)).astype(int)
    # append all test records and start digit recognition
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            # Make sure the suffix is correct and avoid the influence of hidden files
            if os.path.splitext(file)[1] == '.wav':
                test_files.append(os.path.join(root, file))
    for test_file in test_files:
        # get mfcc feature and ignore the warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(test_file)
        # calculate the score and get the maximum score
        sum_prob = []
        for i in range(10):
            prob_i = gmm_models[i].log_prob(features_mfcc)
            sum_prob.append(prob_i)
        pred = str(np.argmax(np.array(sum_prob)))
        ground_truth = os.path.splitext(test_file)[0][-1]
        print("pred: ", pred, "   ground_truth: ", ground_truth)
        if pred == ground_truth:
            pred_true += 1
        confusion_matrix[int(ground_truth), int(pred)] += 1
        count += 1

    print("---------- GMM (GaussianMixture) ----------")
    print("confusion_matrix: \n", confusion_matrix)
    print("Train num: %d, Test num: %d, Predict true num: %d" % (200-count, count, pred_true))
    print("Accuracy: %.2f" % (pred_true * 100 / count))


if __name__ == '__main__':
    train_dir = './processed_train_records'
    test_dir = './processed_test_records'
    hmm_models = train_model_hmm(train_dir, n_components=3, max_iter=100)
    predict_hmm(hmm_models, test_dir)