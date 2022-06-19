# Seasonal Ratio Score (SRS) for Time-Series OOD detection
Python Implementation of RObustTraining forTime-Series (RO-TS) for the paper: "[Out-of-Distribution Detection in Time-Series Domain: A Novel Seasonal Ratio Scoring Approach]()" by Taha Belkhouja, Yan Yan, and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirement.txt
```

## Obtain datasets
- The dataset can be obtained as .zip file from "[The UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php)".
- Download the .zip file and extract it it in `UCRDatasets/{dataset_name}` directory.
- Run the following command for pre-processing a given dataset while specifying if it is multivariate, for example, on SyntheticControl dataset
```
python preprocess_dataset.py --dataset_name=Heartbeat
```
The results will be stored in `Dataset` directory. 

## Run
- Example  training run
```
python --dataset_name Cricket --dataset_name_ood Epilepsy
```
