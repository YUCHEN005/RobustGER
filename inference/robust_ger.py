import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.robust_ger import generate
from lit_gpt import Tokenizer
from lit_gpt.robust_ger import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization
from evaluate import load
wer = load("wer")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
parser.add_argument('--test_data', type=str)
args = parser.parse_args()

devices = args.d

exp_path = f'~/RobustGER/runs/finetune_robust_ger'
predict_dir = f'{exp_path}/predictions_{args.test_data}'  # place to save predictions

data_path = f'~/RobustGER/hypo_paradise/{args.test_data}.pt'

file = 'adapter_best.pth'

precision = None
quantize = None
strategy: str = "auto"
torch.set_float32_matmul_precision("high")

precision = precision or get_default_supported_precision(training=False)
fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
fabric.launch()

dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

checkpoint_dir = Path("~/RobustGER/checkpoints/Llama-2-7b-hf")
check_valid_checkpoint_dir(checkpoint_dir)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

checkpoint_path = checkpoint_dir / "lit_model.pth"

with fabric.init_module(empty_init=True), quantization(quantize):
    model = GPT(config)

tokenizer = Tokenizer(checkpoint_dir)
data = torch.load(data_path)


def result(adapter_path, model):
    # LOADING CORRESPOINDG ADAPTER MODEL
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)

    model.eval()
    model = fabric.setup(model)

    c = 0
    return_dict = {}
    pr = []
    gt = []
    to_json = []
    for datapoint in data:
        encoded = datapoint['input_ids_no_response'].to(model.device)
        emb_diff = datapoint['emb_diff'].to(model.device).to(dtype)
        ground_truth = datapoint['ground_truth']

        max_returned_tokens = encoded.size(0) + 150

        y = generate(
            model=model,
            emb_diff=emb_diff,
            idx=encoded,
            max_returned_tokens=max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=0.2,
            top_k=1,
            eos_id=tokenizer.eos_id
        )

        model.reset_cache()
        output = tokenizer.decode(y)

        inf = output[len(tokenizer.decode(encoded)):].split('\n')[0].strip()
        ref = ground_truth.strip()
        if inf == ref:
            c = c + 1
        pr.append(inf)
        gt.append(ref)
        to_json.append({'inference': inf, 'ground_truth': ref})

    print(f'For {adapter_path}')
    return_dict['adapter_path'] = adapter_path
    wer_ = wer.compute(predictions=pr, references=gt)
    print(f'WER is {wer_}')
    return_dict['WER'] = wer_
    print(f'Ground truth matches is {c}/{len(data)}')
    to_json.append({'wer': wer_, 'gtms': f'{c}/{len(data)}'})
    return_dict['gtms'] = c / len(data)
    os.system(f'mkdir -p {predict_dir}')
    with open(os.path.join(predict_dir, adapter_path.split('/')[-1].split('.pth')[0] + '.json'), 'w') as f:
        f.write(json.dumps(to_json, indent=4, ensure_ascii=False))
    print(os.path.join(predict_dir, adapter_path.split('/')[-2] + '.json'))
    print('the post string normalization wer is')
    x = 0
    for i in range(len(pr)):
        pr[i] = pr[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        gt[i] = gt[i].lower().replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace("'", '')
        if pr[i] == gt[i]:
            x = x + 1
    post_wer = wer.compute(predictions=pr, references=gt)
    print('WER', post_wer)
    return_dict['post_ST_wer'] = post_wer
    print(x, '/', len(pr))
    return_dict['post_gtms'] = x / len(pr)
    print('*********************')
    return return_dict


# file = 'adapter_best.pth'
adapter_path = os.path.join(exp_path, file)

result_dict = result(adapter_path, model)
wer_percent = result_dict['WER'] * 100
wer_percent_post = result_dict['post_ST_wer'] * 100

gt_percent = result_dict['gtms'] * 100
gt_percent_post = result_dict['post_gtms'] * 100

print('epoch: ', file, 'WER: ', wer_percent, "WER_post: ", wer_percent_post, "GTM: ", gt_percent, "GTM_post: ",
      gt_percent_post)

