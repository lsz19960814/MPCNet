# This is the code corresponding to article *"Forecasting turning points in stock price by integrating chart similarity and multipersistence"* submitted to TKDE

## Requirements

- Python (**>=3.6**)
- PyTorch (**>=1.7.1**)
- Pynauty (**=0.6.0**)
- Pyunicorn (**=0.6.1**)
- Networkx (**<=2.3**)
- Other dependencies (pyyaml, easydict)

## Usage

The **Graph_sim** folder corresponds to the **Graph similarity** in the article, and the **Stock** folder corresponds to **Multipersistence GCN**

Due to the upload restrictions of GitHub, we have provided a trained model for both the Chinese and US markets, using data from 2010 to 2021 as the training set and 2022 as the testing set for code validation. Just run train.py

```bash
## Chinese market test
python -u train.py --data_type 2022

## USA market test
python -u train.py --data_type 2022us
```

If retraining is required, you need to first obtain stock daily data from tushare and place it in the day_csv sub-folder under the Graph_sim folder. Run day_sc.py in the Graph_sim folder, then run Run_model.py in the Stock folder, and finally run train.py with the parameter -- need_train

```bash

cd Graph_sim

python day_sc.py

cd Stock

python Run_model.py

## Chinese market train
python -u train.py --data_type 2022 --need_train

## USA market train
python -u train.py --data_type 2022us --need_train
```
