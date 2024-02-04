#!/usr/bin/env bash

source activate <your_conda_env>

# "test_data" specifies the test set, e.g., test_chime4/test_real, test_chime4/test_simu
test_data=test_chime4/test_real
python inference/robust_ger.py --test_data ${test_data}
