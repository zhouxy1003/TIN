# TIN
This is an implementation of the paper [Temporal Interest Network for Click-Through Rate Prediction]().

## Requirements
* Python >= 2.6.1
* NumPy >= 1.12.1
* Pandas >= 0.20.1
* TensorFlow >= 1.4.0 
* GPU with memory >= 10G

## Dataset
* One can refer to the official code of [DIN](https://github.com/zhougr1993/DeepInterestNetwork) to learn about the pre-processing of dataset in detail. Note that the key difference is that in our code, we split the dataset into training, validation, and test set, while in the original code, the authors only use training and test set. Such an approach may introduce **cherry-pick** problems , the performance of the model may be overestimated. Specifically, we save the model with the best performance on the validation set (**GAUC**), and use the prediction result of the saved model on the test set as the evaluation criterion for the current model.

## Training and Evaluation
This implementation contains the method TTA-TR and all the other competitors' method, including DIEN, DSIN, BST, GRU4Rec, SASRec, Bert4Rec. The training procedures of all method is as follows:

* Step 1: Choose a method and enter the model folder.
```
cd model/tin;
```

* Step 2: Run the main program.
```
python train_tin.py;
```
