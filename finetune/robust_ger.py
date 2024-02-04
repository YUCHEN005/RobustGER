import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.wl_nib import GPT, Block, Config, adapter_filter, mark_only_adapter_as_trainable, MINE_v1
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    step_csv_logger,
)
from scripts.prepare_alpaca import generate_prompt
import argparse
import gc

# cli setup
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-3, help='chime4: 1e-2, others: 5e-3')
parser.add_argument('--d', type=int, default=1, help='lNo of GPUs (default: 1)')
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--wp', type=float, default=0.2)
parser.add_argument('--data', type=str)
parser.add_argument('--train_path', type=str)
parser.add_argument('--val_path', type=str)
args = parser.parse_args()

learning_rate = args.lr
learning_rate_m = args.lr * 0.1
dataset = args.data
train_path = args.train_path
val_path = args.val_path

# Hyperparameters
num_epochs = 2
weight_decay = 0.02

# Batch and device stuff
devices = args.d
batch_size = 32 // devices  # trained atis with 32BS 1 gpu == 64BS with 2 GPUs
micro_batch_size = 4  # was 6 with 500
gradient_accumulation_iters = batch_size // micro_batch_size

train_data = torch.load(train_path)#, map_location=torch.device('cpu'))
val_data = torch.load(val_path)#, map_location=torch.device('cpu'))

train_data_len = len(train_data)
val_data_len = len(val_data)

epoch_size = train_data_len // micro_batch_size  # 50000  # train dataset size
max_iters = num_epochs * epoch_size // devices
eval_iters = val_data_len // micro_batch_size // devices  # 100
warmup_steps = int(epoch_size * args.wp) // devices
warmup_steps_m = int(epoch_size * args.wp) // devices

# Network stuff
max_input_length = 1024  # 800 for v100 wo k,v ; 700 works for v100 w k,v

save_interval = 200 // devices
log_interval = 1
# change this value to force a maximum sequence length
override_max_seq_length = None

run_name = f'finetune_{dataset}'  # added z at end to distinguish
out_dir: str = 'runs/' + run_name

input_dim1 = 4096
input_dim2 = 1280

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("~/RobustGER/hypo_paradise"),
    checkpoint_dir: Path = Path("~/RobustGER/checkpoints/Llama-2-7b-hf"),
    out_dir: Path = Path(f"~/RobustGER/runs/{run_name}"),
    precision: Optional[str] = None,
    tpu: bool = False,
):
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    print('precision: ', precision)

    fabric_devices = devices
    if fabric_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            fabric_devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path):
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    config = Config.from_name(name=checkpoint_dir.name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        mine = MINE_v1(input_dim1, input_dim2)

    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to adapter weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_adapter_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    optimizer_m = torch.optim.AdamW(mine.parameters(), lr=learning_rate_m, weight_decay=weight_decay)
    mine, optimizer_m = fabric.setup(mine, optimizer_m)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, mine, optimizer_m, train_data, val_data, checkpoint_dir, out_dir, speed_monitor)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_adapter_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    mine,
    optimizer_m: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)

    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)

    max_seq_length = min(max_seq_length, max_input_length)
    longest_seq_length = min(longest_seq_length, max_input_length)

    # sanity check
    validate(fabric, model, val_data, tokenizer, longest_seq_length)

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    for iter_num in range(max_iters // gradient_accumulation_iters):

        micro_batch = []
        for iter in range(gradient_accumulation_iters):
            real_iter = iter_num * gradient_accumulation_iters + iter
            input_ids, targets, emb_diff, audio_features, clean_audio_features = get_batch_para(
                fabric, model, train_data, longest_seq_length, longest_seq_ix if real_iter == 0 else None
            )
            micro_batch.append((input_ids, targets, emb_diff, audio_features, clean_audio_features))


        #### step 1: mine
        for iter in range(gradient_accumulation_iters):

            real_iter = iter_num * gradient_accumulation_iters + iter
            if real_iter <= warmup_steps_m:
                lr = learning_rate_m * real_iter / warmup_steps_m  # what is happening here
                for param_group in optimizer_m.param_groups:
                    param_group['lr'] = lr

            input_ids, targets, emb_diff, audio_features, clean_audio_features = micro_batch[iter]

            if iter == 0:
                mark_as_trainable(mine)
                mark_as_untrainable(model)
                optimizer_m.zero_grad()
                optimizer.zero_grad()

            t0 = time.time()
            with torch.no_grad():
                _, emb_diff_mids = model(input_ids, emb_diff=emb_diff, max_seq_length=max_seq_length, lm_head_chunk_size=128)

            with fabric.no_backward_sync(mine, enabled=(iter + 1) % gradient_accumulation_iters != 0):

                loss_m = -mine(-emb_diff.detach(), audio_features.detach(), raw=True).mean() + \
                         mine(-emb_diff.detach(), clean_audio_features.detach(), raw=True).mean()

                fabric.backward(loss_m / gradient_accumulation_iters)

            if (iter + 1) % gradient_accumulation_iters == 0:
                mark_as_trainable(mine)
                mark_as_untrainable(model)
                optimizer_m.step()
                optimizer_m.zero_grad()
                optimizer.zero_grad()

            dt = time.time() - t0
            if real_iter % log_interval == 0:
                fabric.print(f"iter {real_iter + 1} step1: loss_m {loss_m.item():.4f}, time: {dt:.2f}s")

        ### step 2: llama + mine
        for iter in range(gradient_accumulation_iters):

            real_iter = iter_num * gradient_accumulation_iters + iter
            if real_iter <= warmup_steps:
                lr = learning_rate * real_iter / warmup_steps  # what is happening here
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            input_ids, targets, emb_diff, audio_features, clean_audio_features = micro_batch[iter]

            if iter == 0:
                mark_as_trainable(mine)
                mark_only_adapter_as_trainable(model)
                optimizer.zero_grad()
                optimizer_m.zero_grad()

            t0 = time.time()
            with fabric.no_backward_sync(model, enabled=(iter + 1) % gradient_accumulation_iters != 0):
                logits, emb_diff_mids = model(input_ids, emb_diff=emb_diff, max_seq_length=max_seq_length, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss_ce = chunked_cross_entropy(logits, targets[..., 1:])

                loss_m = []
                for emb_diff_mid in emb_diff_mids:
                    loss_m.append(-mine(-emb_diff_mid, audio_features.detach(), raw=False).mean())
                loss_m = sum(loss_m) / len(loss_m)

                loss = loss_ce + loss_m * args.lamda
                fabric.backward(loss / gradient_accumulation_iters)

            if (iter + 1) % gradient_accumulation_iters == 0:
                mark_as_untrainable(mine)
                mark_only_adapter_as_trainable(model)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_m.zero_grad()

            dt = time.time() - t0
            if real_iter % log_interval == 0:
                fabric.print(f"iter {real_iter + 1} step2: loss {loss.item():.4f}, loss_ce {loss_ce.item():.4f}, loss_m {loss_m.item():.4f}, time: {dt:.2f}s")

            if (real_iter + 1) % save_interval == 0:
                checkpoint_path = out_dir / f"iter-{(real_iter + 1):06d}.pth"
                save_adapter_checkpoint(fabric, model, checkpoint_path)

                val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
                fabric.print(f"step {(real_iter + 1)}: val loss {val_loss:.4f}")
                fabric.barrier()
                print('End of iters ', (real_iter + 1) + 1)


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets, emb_diff = get_batch(fabric, model, val_data, longest_seq_length)
        # logits = model(input_ids)
        logits, emb_diff_mids = model(input_ids, emb_diff=emb_diff)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        losses[k] = loss.item()
    val_loss = losses.mean()

    model.reset_cache()
    model.train()
    return val_loss.item()


def get_batch(
    fabric: L.Fabric, model, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"][:max_input_length].type(torch.int64) for i in ix]
    labels = [data[i]["labels"][:max_input_length].type(torch.int64) for i in ix]
    emb_diff = [data[i]["emb_diff"].type(model.dtype) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = max(len(s) for s in input_ids) if fabric.device.type != "xla" else longest_seq_length

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    ef = torch.stack([x for x in emb_diff], dim=0)

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y, ef = fabric.to_device((x.pin_memory(), y.pin_memory(), ef.pin_memory()))
    else:
        x, y, ef = fabric.to_device((x, y, ef))
    return x, y, ef


def get_batch_para(
        fabric: L.Fabric, model, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"][:max_input_length].type(torch.int64) for i in ix]
    labels = [data[i]["labels"][:max_input_length].type(torch.int64) for i in ix]
    emb_diff = [data[i]["emb_diff"].type(model.dtype) for i in ix]
    audio_features = [data[i]["audio_features"].type(model.dtype) for i in ix]
    clean_audio_features = [data[i]["clean_audio_features"].type(model.dtype) for i in ix]

    # it's better to pad to a fixed seq length with XLA to avoid recompilation
    max_len = max(len(s) for s in input_ids) if fabric.device.type != "xla" else longest_seq_length

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    ef = torch.stack([x for x in emb_diff], dim=0)
    af = torch.stack([x for x in audio_features], dim=0)
    c_af = torch.stack([x for x in clean_audio_features], dim=0)

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y, ef, af, c_af = fabric.to_device((x.pin_memory(), y.pin_memory(), ef.pin_memory(), af.pin_memory(), c_af.pin_memory()))
    else:
        x, y, ef, af, c_af = fabric.to_device((x, y, ef, af, c_af))
    return x, y, ef, af, c_af


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    print(f'mean length = {sum(lengths) / len(lengths)}')
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_adapter_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving adapter weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


def mark_as_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def mark_as_untrainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
