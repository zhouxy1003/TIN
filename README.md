# TIN
This is an implementation of the paper [Temporal Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/2308.08487.pdf).

## Requirements
- Python >= 2.6.1
- NumPy >= 1.12.1
- Pandas >= 0.20.1
- TensorFlow >= 1.4.0 
- GPU with memory >= 10G

## Download dataset and preprocess
One can refer to the official code of [DIN](https://github.com/zhougr1993/DeepInterestNetwork) to learn about the pre-processing of the amazon product dataset in detail in the **Download dataset and preprocess** section. 

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

## Training and evaluation
This implementation contains the method TIN and all the other competitors, including DIN, DIEN, DSIN, BST, GRU4Rec, SASRec, Bert4Rec, etc. The training procedures of all method is as follows:

- Step 1: Choose a method and enter the model folder (Take TIN as an example).
    ```bash
    cd model/tin
    ```

- Step 2: Modify the dataset path and run the main program.
    ```bash
    python train_tin.py
    ```

## Measurement of learned quadruple correlation

### Visualize the category-wise target-aware correlation (CTC)
- Step 1: Preprocess the dataset and save the statistics for computing the category-wise target-aware correlation.
    ```bash
    cd visualization
    python amazon_pre_pos.py
    ```
- Step 2: Compute and plot the category-wise target-aware correlation.
    ```bash
    python amazon_mul_c_pos.py
    ```

<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/visualization/amazon_category_674.png" alt="CTC" width="50%"></div>

### Visualize the learned quadruple correlation
- Step 1: Save the value of position embedding and category embedding of different models after training as `p.npy` and `c.npy`.
- Step 2: Compute the learned quadruple correlation.
    ```bash
    python temporal_correlation.py
    ```
<div align=center><img src="https://github.com/zhouxy1003/TIN/blob/main/visualization/amazon_category_674.png" alt="learned quadruple correlation of TIN" width="50%"></div>
