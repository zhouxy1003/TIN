# Temporal Interest Network for User Response Prediction

The open source code for WWW 2024 paper "[Temporal Interest Network for User Response Prediction](https://arxiv.org/abs/2308.08487)".

## Model overview
### TIN architecture
<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/model/TIN.png" alt="TIN architecture" width="50%"></div>

### Temporal Interest Module
<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/model/TIN_module.png" alt="Temporal Interest Module" width="50%"></div>

## Getting started

### Requirements
- Python >= 2.6.1
- NumPy >= 1.12.1
- Pandas >= 0.20.1
- TensorFlow >= 1.4.0 
- GPU with memory >= 10G

### Download dataset and preprocess
One can refer to the official code of [DIN](https://github.com/zhougr1993/DeepInterestNetwork) to learn about the pre-processing of the Amazon product dataset in detail in the **Download dataset and preprocess** section. 

- Step 1: Download the [Amazon product dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) of Electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
    ```bash
    mkdir raw_data/
    cd data
    bash 0_download_raw.sh
    ```

- Step 2: Convert raw data to pandas dataframe, and remap categorical id.
    ```bash
    python 1_convert_pd.py
    python 2_remap_id.py
    ```

- Step 3: Build the amazon product dataset.
    ```bash
    python build_dataset.py
    ```
Note that the key difference is that in our code, we split the dataset into training, validation, and test set, while in the original code of DIN, the authors only use training and test set. Such an approach may introduce **cherry-pick** problems, the performance of the model may be overestimated. Specifically, we save the model with the best performance on the validation set (**GAUC**), and use the prediction result of the saved model on the test set as the evaluation criterion for the current model.

### Training and evaluation
This implementation contains the method TIN and all the other competitors, including DIN, DIEN, DSIN, BST, GRU4Rec, SASRec, Bert4Rec, etc. The training procedures of all method is as follows:

- Step 1: Choose a method and enter the model folder (Take TIN as an example).
    ```bash
    cd model/tin
    ```

- Step 2: Modify the dataset path and train a TIN model by running the following script.
    ```bash
    python train_tin.py
    ```

## Measurement of semantic-temporal correlation

### Visualize the Category-wise Target-aware Correlation (CTC)
- Step 1: Preprocess the dataset and save the statistics for computing the Category-wise Target-aware Correlation.
    ```bash
    cd visualization
    python amazon_pre_pos.py
    ```
- Step 2: Compute and plot the Category-wise Target-aware Correlation.
    ```bash
    python amazon_mul_c_pos.py
    ```

<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/visualization/ground_truth_STC.png" alt="Category-wise Target-aware Correlation" width="50%"></div>

### Visualize the learned semantic-temporal correlation
- Step 1: Save the value of position embedding and category embedding of different models after training as `p.npy` and `c.npy`.
- Step 2: Compute the learned quadruple correlation.
    ```bash
    python temporal_correlation.py
    ```
    
<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/visualization/TIN_STC.png" alt="Learned semantic-temporal correlation of TIN" width="50%"></div>

## Concat
If you have any question about this implementation, please create an issue or send us an Email at:
- zhouxy1003@163.com (Xinyi Zhou)

If you have any question about the model and the paper, feel free to contact:
- jonaspan@tencent.com (Junwei Pan)

## Citation
If you find our code or propcessed data helpful in your research, please kindly cite the following papers.
```bibtex
@article{zhou2023temporal,
  title={Temporal interest network for user response prediction},
  author={Zhou, Haolin and Pan, Junwei and Zhou, Xinyi and Chen, Xihua and Jiang, Jie and Gao, Xiaofeng and Chen, Guihai},
  booktitle={Proceedings of the 2024 World Wide Web Conference},
  year={2024}
}
```
