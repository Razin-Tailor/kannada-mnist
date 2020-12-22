# kannada-mnist 
A Resnet34 solution to the [Kaggle Challenge](https://www.kaggle.com/c/Kannada-MNIST/)

This repo shows a basic usage of PyTorch to create a training and inference pipeline for Image Classification Task.

This approach achieved the public score of 97%

Follow the following steps to successfully utilize the repo

## Environment Setup

- Create a virtual environment and activate it
- Use the following command to install the project dependencies `pip install -r requirements.txt`

## Data Creation

- First is creation of folds. Folding basically provides train/test indices to split data in train/test sets.
- For fold creation [Stratified KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) is used
- Image data is 28 * 28 and it is flattened out in rows in `train.csv`
- Reading from dataframe while training is slow, Hence, I create pickle objects for the image data in the CSV
- `python create_image_pickles.py` This will create image pickles which we will use to train the model

## Training

- There are some default params set in `train.py`. Feel free to tweak them.
- To train the resnet34 model, run `python train.py` and it should start training

## Inference and Submission

Please refer my [Kaggle Kernel](https://www.kaggle.com/razintailor/kannada-mnist-resnet34) for the inference pipeline
