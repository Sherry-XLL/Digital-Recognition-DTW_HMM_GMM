## Digital recognition
#### Copyright © Sherry-XLL. All rights reserved.

> Github 项目地址：[https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM][1]

> CSDN 文档地址：[https://blog.csdn.net/Sherry_ling/article/details/118713802][13]

在孤立词语音识别(Isolated Word Speech Recognition) 中，**DTW**，**GMM** 和 **HMM** 是三种典型的方法：

- **动态时间规整(DTW, Dyanmic Time Warping)** 
- **高斯混合模型(GMM, Gaussian Mixed Model)** 
- **隐马尔可夫模型(HMM, Hidden Markov Model)**

本项目并不介绍这三种方法的基本原理，而是**侧重于 Python 版代码的实现**，针对一个具体的语音识别任务——10 digits recognition system，分别使用 DTW、GMM 和 HMM 建立对 0～9 十个数字的孤立词语音分类识别模型。详细内容可以参考本人的 CSDN 文档 [【SLP·Python】基于 DTW GMM HMM 三种方法实现的语音分类识别系统][13]。

## Experimental Environment
```
  macOS Catalina Version 10.15.6, Python 3.8, PyCharm 2020.2 (Professional Edition).
```

## Introduction
```
  dtw.py: Implementation of Dynamic Time Warping (DTW)
  gmm.py: Implementation of Gaussian Mixture Model (GMM)
  hmm.py: Implementation of Hidden Markov Model (HMM)

  gmm_from_sklearn.py: Train gmm model with GaussianMixture from sklearn
  hmm_from_hmmlearn.py: Train hmm model with hmm from hmmlearn

  preprocess.py: preprocess audios and split data
  processed_test_records: records with test audios
  processed_train_records: records with train audios
  records: original audios
  utils.py: utils function
```

## Launch the script
```
  eg:
  python preprocess.py (mkdir processed records)
  python dtw.py 
```

## Results
各个方法的数据集和预处理部分完全相同，下面是运行不同文件的结果：

```
python dtw.py
----------Dynamic Time Warping (DTW)----------
Train num: 160, Test num: 40, Predict true num: 31
Accuracy: 0.78
```

```
python gmm_from_sklearn.py:
---------- GMM (GaussianMixture) ----------
Train num: 160, Test num: 40, Predict true num: 34
Accuracy: 0.85
```

```
python gmm.py
---------- Gaussian Mixture Model (GMM) ----------
confusion_matrix: 
 [[4 0 0 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 0 0 0 0]
 [0 0 4 0 0 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0 0]
 [0 0 0 0 4 0 0 0 0 0]
 [0 0 0 0 0 4 0 0 0 0]
 [0 0 0 0 0 0 4 0 0 0]
 [0 0 0 0 0 0 0 4 0 0]
 [0 0 0 0 0 0 0 0 4 0]
 [0 0 0 0 0 0 0 0 0 4]]
Train num: 160, Test num: 40, Predict true num: 40
Accuracy: 1.00
```

```
python hmm_from_hmmlearn.py
---------- HMM (GaussianHMM) ----------
Train num: 160, Test num: 40, Predict true num: 36
Accuracy: 0.90
```

```
python hmm.py
---------- HMM (Hidden Markov Model) ----------
confusion_matrix: 
 [[4 0 0 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 0 0 0 0]
 [0 0 3 0 1 0 0 0 0 0]
 [0 0 0 4 0 0 0 0 0 0]
 [0 0 0 0 4 0 0 0 0 0]
 [0 0 0 0 1 3 0 0 0 0]
 [0 0 0 0 0 0 3 0 1 0]
 [0 0 0 0 0 0 0 4 0 0]
 [0 0 0 1 0 0 1 0 2 0]
 [0 0 0 0 0 0 0 0 0 4]]
Train num: 160, Test num: 40, Predict true num: 35
Accuracy: 0.875
```


| Method | DTW | GMM from sklearn | Our GMM | HMM from hmmlearn | Our HMM |
|--|--|--|--|--|--|
| Accuracy | 0.78 | 0.85 | 1.00 | 0.90 | 0.875 |

值得注意的是，上面所得正确率仅供参考。我们阅读[源码][1]就会发现，不同文件中 `n_components` 的数目并不相同，最大迭代次数 `max_iter` 也会影响结果。设置不同的超参数 (hyper parameter) 可以得到不同的正确率，上表并不是各种方法的客观对比。事实上 [scikit-learn][7] 的实现更完整详细，效果也更好，文中三种方法的实现仅为基础版本。

## Contributing

Please let me know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM/issues).

Thanks for insightful suggestions from [@Nian-Chen](https://github.com/Nian-Chen).


## Reference Resources
 1. [https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM][1]
 2. [https://github.com/rocketeerli/Computer-VisionandAudio-Lab/tree/master/lab1][2]
 3. [http://librosa.github.io/librosa/core.html][3]
 4. [https://python-speech-features.readthedocs.io/][4]
 5. [http://yann.lecun.com/exdb/mnist/][5]
 6. [https://www.scikitlearn.com.cn/0.21.3/20/#211][6]
 7. [https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge][7]
 8. [https://blog.csdn.net/nsh119/article/details/79584629?spm=1001.2014.3001.5501][8]
 9. [https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py][9]
 10. [https://github.com/hmmlearn/hmmlearn][10]
 11. [https://hmmlearn.readthedocs.io/en/stable][11]
 12. [https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn-hmm][12]
 13. [https://blog.csdn.net/Sherry_ling/article/details/118713802][13]

[1]: https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM
[2]: https://github.com/rocketeerli/Computer-VisionandAudio-Lab/tree/master/lab1
[3]: http://librosa.github.io/librosa/core.html
[4]: https://python-speech-features.readthedocs.io/
[5]: http://yann.lecun.com/exdb/mnist/
[6]: https://www.scikitlearn.com.cn/0.21.3/20/#211
[7]: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge
[8]: https://blog.csdn.net/nsh119/article/details/79584629?spm=1001.2014.3001.5501
[9]: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py
[10]: https://github.com/hmmlearn/hmmlearn
[11]: https://hmmlearn.readthedocs.io/en/stable
[12]: https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn-hmm
[13]: https://blog.csdn.net/Sherry_ling/article/details/118713802
