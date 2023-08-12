# EAMNet: An Alzheimer's Disease Prediction Model Based on Representation Learning.
Usage Instructions for the Code:
1. Input your dataset directory path into the designated location in `train.py`. The program will utilize `dataset.py` to convert the dataset format into RGB three-channel images. The dataset will be divided into training set, validation set, and test set in the ratio of 0.8:0.1:0.1. Each folder will contain AD, MCI, and NC categories.
2. `accquire_mean_std.py` is used to calculate the mean and standard deviation of the dataset, which will be used for data normalization in both the training and testing code.
3. Running `train.py` will generate a parameter file `.pth`. In `test.py`, add the parameter file path to the specified location. This will result in the generation of a confusion matrix and accuracy values for the test results.
