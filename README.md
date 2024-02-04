# Large Language Models are Efficient Learners of Noise-Robust Speech Recognition

## Conda Environment Configuration

Our code is built based on [lit-gpt](https://github.com/Lightning-AI/lit-gpt/tree/main), please refer to original repo to build the conda environment.

## Code

- Model code: `lit_gpt/robust_ger.py`;
- Training script: `finetune.sh`;
- Inference script: `infer.sh`;

To run the training or inference script, you need to enter the scripts and modify the absolute paths of data, model, and experiment directory. Then, directly run the `.sh` script using `bash` command.

## Models

- For LLMs, please refer to [tutorial](https://github.com/Lightning-AI/lit-gpt/tree/main/tutorials) for details, which support many mainstream LLMs like LLaMA-2;
- For trained adapter weights, please refer to our [HuggingFace repo](https://huggingface.co/PeacefulData/RobustGER).

## Dataset

We have released our Robust HyPoradise dataset at [HuggingFace](https://huggingface.co/datasets/PeacefulData/Robust-HyPoradise).

## Reference
```bib
@article{hu2024large,
  title={Large Language Models are Efficient Learners of Noise-Robust Speech Recognition},
  author={Hu, Yuchen and Chen, Chen and Yang, Chao-Han Huck and Li, Ruizhe and Zhang, Chao and Chen, Pin-Yu and Chng, EnSiong},
  journal={arXiv preprint arXiv:2401.10446},
  year={2024}
}
```