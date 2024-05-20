# This is the code corresponding to article *"Forecasting turning points in stock price by integrating chart similarity and multipersistence"* submitted to TKDE. 

## Requirements

- Python (**>=3.6**)
- PyTorch (**>=1.1.0**)
- Tensorboard(**>=1.4.0**) (for ***visualization***)
- Other dependencies (pyyaml, easydict)

## Usage 

The *Graph_sim* folder corresponds to the Graph similarity in the article, and the Stock folder corresponds to Multipersistence GCN

Due to the upload limitation of GitHub, we provide a trained model with data from 2010 to 2021 as the training set and data from 2022 as the test set as the code validation. Just run train.py

```bash
## Chinese market test
python -u train.py --data_type 2022

## USA market test
python -u train.py --data_type 2022us


If retraining is required, you need to first obtain stock daily data from tushare and place it in the day_csv sub folder under the Graphssim folder. Run day_sc.py in the Graph_sim folder, then run Run_model. py in the Stock folder, and finally run train.py with the parameter -- need_train

## Chinese market train
python -u train.py --data_type 2022 --need_train

## USA market train
python -u train.py --data_type 2022us --need_train
``` 
