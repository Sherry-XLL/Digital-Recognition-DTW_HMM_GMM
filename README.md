## Digital recognition
#### Copyright © Sherry-XLL. All rights reserved.
10 digits recognition system based on DTW, HMM and GMM.

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

## Contributing

Please let me know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/Sherry-XLL/Digital-Recognition-DTW_HMM_GMM/issues).

Thanks for insightful suggestions from [@Nian-Chen](https://github.com/Nian-Chen).


## Reference Resources
```
  preprocess.py referencing 
  https://github.com/rocketeerli/Computer-VisionandAudio-Lab/tree/master/lab1
```
