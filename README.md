# This repo contains the code for the final project of UCSD DSC 291

## We explore different pretraining strategies and their effects to robustness of a CNN for pathology classification
- We consider three sets of weights (1) Barlow Twins pretrained weights, (2) Image-net pretrained weights, and (3) random initialization (no pretraining)
- We used resnet34 as the backbone network.

## Dataset 
- We used the pathmnist dataset, which is a sub-dataset of the [medmnist](https://medmnist.com/). To be able to run our code, please download the data from [zenodo](https://zenodo.org/record/5208230)
and select the pathmnist file.

## Experiment Setup
- We split the training dataset randomly into two, one called train_set, and the other called unlabel_set. The unlabel set contains 90% of the original training dataset, in which
we use the ONLY the images for self-supervised pretraining purposes. For all the different weight initialization methods, we only used 10% of the original training dataset to finetune.
In addition, we choose the learning rate by sweeping through : lr = {1e-5,1e-4,1e-3,1e-2,1e-1,5e-1}, wd = {0,1e-6,1e-5,1e-4,1e-3}

## Code Organization
- bt_exp folder contains all the code to pretrain a resnet34 using (Barlow Twins)[https://arxiv.org/pdf/2103.03230.pdf]. First run pretrain.py to pretrain, then run 
sweep.py to find the optimal hyperparameters for finetuning. Using the best hyper-parameters, run finetune.py

- supervised_exp contains the code for both training from scratch and imagenet pretraining. For the image-net case, run sweep.py followed by finetune.py. For fully_sup, run sweep.py 
then train.py

## Results:
| Pretrain Method | Test Accuracy on Clean Images (%)|
| --------------- | ----------------- |
| None | 70.1 |
| Barlow Twin | 74.8 |
| ImageNet | 82.9 |
