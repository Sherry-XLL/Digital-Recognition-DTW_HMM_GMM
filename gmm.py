# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import librosa


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
    :param means: The mean of each mixture component.
    :param var: The variance of each mixture component.
    :return: the log Gaussian probability
    """
    return (-0.5 * np.log(var) - np.divide(np.square(x - means), 2 * var) - 0.5 * np.log(2 * np.pi)).sum()


class GMM:
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    mfcc_data : array, shape (, 39)
        Mfcc data of training wavs.

    random_state: int
        RandomState instance or None, optional (default=0)

    Attributes
    ----------
    means : array, shape (n_components, 1)
        Mean parameters for each mixture component in each state.

    var : array, shape (n_components, 1)
        Variance parameters for each mixture component in each state.

    weights : array, shape (1, n_components)
        Weights matrix of each component.

    """
    def __init__(self, mfcc_data, n_components, random_state=0):
        # Initialization
        self.mfcc_data = mfcc_data
        self.means = np.tile(np.mean(self.mfcc_data, axis=0), (n_components, 1))
        # randomization
        np.random.seed(random_state)
        for k in range(n_components):
            randn_k = np.random.randn()
            self.means[k] += 0.01 * randn_k * np.sqrt(np.var(self.mfcc_data, axis=0))
        self.var = np.tile(np.var(self.mfcc_data, axis=0), (n_components, 1))
        self.weights = np.ones(n_components) / n_components
        self.n_components = n_components

    def e_step(self, x):
        """
        Expectation-step.
        :param x: input array-like data
        :return: Logarithm of the posterior probabilities (or responsibilities) of
                 the point of each sample in x.
        """
        log_resp = np.zeros((x.shape[0], self.n_components))
        for i in range(x.shape[0]):
            log_resp_i = np.log(self.weights)
            for j in range(self.n_components):
                log_resp_i[j] += log_gaussian_prob(x[i], self.means[j], self.var[j])
            y = np.exp(log_resp_i - log_resp_i.max())
            log_resp[i] = y / y.sum()
        return log_resp

    def m_step(self, x, log_resp):
        """
        Maximization step.
        :param x: input array-like data
        :param log_resp: Logarithm of the posterior probabilities (or responsibilities) of
                      the point of each sample in data
        """
        self.weights = np.sum(log_resp, axis=0) / np.sum(log_resp)
        denominator = np.sum(log_resp, axis=0, keepdims=True).T
        means_num = np.zeros_like(self.means)
        for k in range(self.n_components):
            means_num[k] = np.sum(np.multiply(x, np.expand_dims(log_resp[:, k], axis=1)), axis=0)
        self.means = np.divide(means_num, denominator)
        var_num = np.zeros_like(self.var)
        for k in range(self.n_components):
            var_num[k] = np.sum(np.multiply(np.square(np.subtract(x, self.means[k])),
                                            np.expand_dims(log_resp[:, k], axis=1)), axis=0)
        self.var = np.divide(var_num, denominator)

    def train(self, x):
        """
        :param x: input array-like data
        """
        log_resp = self.e_step(x)
        self.m_step(x, log_resp)

    def log_prob(self, x):
        """
        :param x: input array-like data
        :return: calculated log gaussian probability of single modal Gaussian
        """
        sum_prob = 0
        for i in range(x.shape[0]):
            prob_i = np.array([np.log(self.weights[j]) + log_gaussian_prob(x[i], self.means[j], self.var[j])
                               for j in range(self.n_components)])
            sum_prob += np.max(prob_i)
        return sum_prob


def train_model_gmm(train_dir, n_components, max_iter=100, random_state=0):
    """
    :param train_dir: training directory
    :param n_components: The number of mixture components
    :param max_iter: The number of EM iterations to perform
    :return: gmm_models
    """
    gmm_models = []
    mfcc_data = get_mfcc_data(train_dir)
    for i in range(10):
        gmm_models.append(GMM(mfcc_data[i], n_components=n_components, random_state=random_state))
    iter = 1
    lower_bound = -np.infty
    while iter <= max_iter:
        prev_lower_bound = lower_bound
        total_log_like = 0.0
        for i in range(10):
            gmm_models[i].train(mfcc_data[i])
            total_log_like += gmm_models[i].log_prob(mfcc_data[i])
        lower_bound = total_log_like
        if abs(lower_bound - prev_lower_bound) < 1:
            break
        print("Iteration:", iter, " log_prob:", lower_bound)
        iter += 1

    return gmm_models


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

    print("---------- Gaussian Mixture Model (GMM) ----------")
    print("confusion_matrix: \n", confusion_matrix)
    print("Train num: %d, Test num: %d, Predict true num: %d" % (200-count, count, pred_true))
    print("Accuracy: %.2f" % (pred_true * 100 / count))


if __name__ == '__main__':
    train_dir = './processed_train_records'
    test_dir = './processed_test_records'
    gmm_models = train_model_gmm(train_dir, n_components=6)
    predict_gmm(gmm_models, test_dir)