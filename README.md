## Digital recognition
#### Copyright © Sherry-XLL. All rights reserved.
10 digits recognition system based on DTW, HMM and GMM.

## Introduction
```
  dtw.py: Dynamic Time Warping (DTW)
  hmm.py: Train hmm model
  gmm.py: Train gmm model
  preprocess.py: preprocess audios and split data
  processed_test_records: records with test audios
  processed_train_records: records with train audios
  records: original audios
  utils.py: utils function
```

## Launch the script
```
  Move records folder to the same directory
  python preprocess.py (mkdir processed records)
  python dtw.py 
  python hmm.py 
  python gmm.py 
```

## Result
```
  Num of train: 160, num of test: 40
  Accuracy of dtw: 30/40 78%
  Accuracy of hmm: 36/40 90%
  Accuracy of gmm: 35/40 88%
```

## Reference Resources
```
  https://github.com/rocketeerli/Computer-VisionandAudio-Lab/tree/master/lab3
```
