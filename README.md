<div align="center">

# The code of the paper presenting FMDST model accepted by ICIC2026.

</div>

## Description

In this paper, we propose a deep learning framework, FMDST model, which leverages spatio-temporal correlations among different sites to achieve ultra-short-term multi-step ahead forecasting for multi-site wind power.

![My Image](overall.png)

## Project Structure
```
FMDST/
├── Ablation_model/                                     # Contains all ablation models.
├── Data_Preprocessing/                                 # Scripts for data processing and dataset creation.
│   ├── data_clean/                                     # Cleaning missing and abnormal values.
│   ├── AVG/                                            # Calculate the average of multiple results.
│   ├── create_miss_data/                               # Create missing data.
│   ├── data_preprocess/                                # Pre-generation of train/val/test input sequences.
│   └── MIC/                                            # feature selection and visualization using MIC.
├── model/                                              # Implementation of the proposed and baseline models.
├── model_save/                                         # Saved model files.
├── SDWPF/                                              # ST_ReP data.
├── sdwpf_kddcup/                                       # Dataset files.
│   ├── sdwpf_245days_v1.csv/                           # Raw wind power data.
│   ├── sdwpf_baidukddcup2022_turb_location.csv/        # Relative locations of all wind turbines.
│   ├── unclean_data.csv/                               # Linear interpolation was used to impute both missing and abnormal values.
│   └── Cleaned_data_with_power.csv/                    # Data with missing and outlier values corrected.
├── ST_ReP/                                             # ST-ReP model and data preprocessing and running functions.
├── Time_Series_Library_main.py                         # TSLib is an open-source library including Informer and PatchTST.
├── main.py                                             # Main entry point.
├── train_evaluate.py                                   # Code for model training and evaluation.
└── README.md                                           # Project documentation.
```                                      # Project documentation.

## Dataset

The dataset used in this project is **SDWPF (Spatial Dynamic Wind Power Forecasting)**, which was released for the **Baidu KDD Cup 2022** challenge.  You can view and download the dataset [here](https://aistudio.baidu.com/competition/detail/152/0/introduction) after registration.

