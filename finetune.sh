#!/usr/bin/env bash

source activate <your_conda_env>

# "data" specifies the dataset name
# "train_path" specifies the training data path
# "val_path" specifies the valid data path
data=chime4
python finetune/robust_ger.py --data ${data} \
       --train_path <your_data_path>/train_${data}.pt \
       --val_path <your_data_path>/val_${data}.pt \
