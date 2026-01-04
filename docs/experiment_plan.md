# Experiment plan

## Sanity checks

Quick correctness and profiling runs (around 10 minutes):

1) Debug partition, tiny model, fast val

```bash
foundationts data download --partition-set debug
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 32 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 64 \
  --n-decoder-layers 2 \
  --num-experts 2 \
  --num-expert-layers 1 \
  --k 1 \
  --n-head 2 \
  --val-split 0.01 \
  --val-every 50 \
  --val-max-batches 2 \
  --log-every 10 \
  --checkpoint-every 0
```

2) OOD validation sanity check (finance)

```bash
foundationts train \
  --dataset-path time300b_selected \
  --ood-val-dataset-path time300b_selected/finance \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 32 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 64 \
  --n-decoder-layers 2 \
  --num-experts 2 \
  --num-expert-layers 1 \
  --k 1 \
  --n-head 2 \
  --val-split 0.01 \
  --val-every 50 \
  --val-max-batches 2 \
  --ood-val-max-batches 2 \
  --log-every 10 \
  --checkpoint-every 0
```

## Single-GPU capacity + architecture probes

Goal: find the largest batch/model that fits on one GPU and decide whether pointwise or patch tokenization
is more efficient for the same model size. Use the same config for both and sweep `--batch-size` until OOM.

Model target (roughly TIME-MOE base scale for a single GPU probe):
- hidden_size 384, n_decoder_layers 12, num_experts 8, num_expert_layers 1, k 2, n_head 12

Paper reference sizes (for scale intuition):
- base: 12L/12H/8E k=2 dmodel=384 dff=1536 dexpert=192 (50M active / 113M total)
- large: 12L/12H/8E k=2 dmodel=768 dff=3072 dexpert=384 (200M active / 453M total)
- ultra: 36L/16H/8E k=2 dmodel=1024 dff=4096 dexpert=512 (1.1B active / 2.4B total)

Pointwise tokens (no patch):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 128 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 12 \
  --val-split 0 \
  --log-every 10 \
  --checkpoint-every 0
```

Patch tokens (same model, `patch_len=32`):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 128 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 12 \
  --patch \
  --patch-len 32 \
  --patch-stride 32 \
  --val-split 0 \
  --log-every 10 \
  --checkpoint-every 0
```

If both fit comfortably, increase `--batch-size` (or `--hidden-size`) and re-run until OOM. Use the new
param count printout to track total vs active params as you scale.

## Real experiments

1) Full train partitions, mid-size model, regular checkpoints

```bash
foundationts data download --partition-set train
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 10000 \
  --epochs 1 \
  --batch-size 256 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 256 \
  --n-decoder-layers 4 \
  --num-experts 4 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 8 \
  --val-split 0.01 \
  --val-every 1000 \
  --val-max-batches 10 \
  --checkpoint-every 2000
```

2) Longer run with OOD validation

```bash
foundationts train \
  --dataset-path time300b_selected \
  --ood-val-dataset-path time300b_selected/finance \
  --steps-per-epoch 10000 \
  --epochs 3 \
  --batch-size 256 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 256 \
  --n-decoder-layers 4 \
  --num-experts 4 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 8 \
  --val-split 0.01 \
  --val-every 1000 \
  --val-max-batches 10 \
  --ood-val-max-batches 10 \
  --checkpoint-every 2000
```

Notes:

- The Time-300B downloader uses `huggingface_hub.snapshot_download`; set
  `HUGGINGFACE_HUB_TOKEN` if your environment requires auth.
- `build_ts_dataset` can load a single file or a directory of supported
  datasets. See `docs/data.md`.



## Notes

### Testing pointwise small

I'm getting some nans using the following command

```
foundationts train \
--dataset-path time300b_selected \
--steps-per-epoch 200 \
--epochs 1 \
--batch-size 128 \
--seq-max-len 512 \
--seq-stride 512 \
--hidden-size 384 \
--n-decoder-layers 12 \
--num-experts 8 \
--num-expert-layers 1 \
--k 2 \
--n-head 12 \
--val-split 0 \
--log-every 10 \
--checkpoint-every 0 
params total=23.11M (23,108,841) active=12.49M (12,492,009)
step=10 loss=3.6159 pred=0.5888 aux=151.3556 lr=1.00e-06 toks/s=23,243
step=20 loss=3.5834 pred=0.5931 aux=149.5184 lr=2.00e-06 toks/s=69,733
step=30 loss=3.5841 pred=0.6058 aux=148.9136 lr=3.00e-06 toks/s=72,855
step=40 loss=3.4019 pred=0.5189 aux=144.1524 lr=4.00e-06 toks/s=75,907
step=50 loss=3.3548 pred=0.4704 aux=144.2183 lr=5.00e-06 toks/s=77,377
step=60 loss=3.3950 pred=0.4846 aux=145.5180 lr=6.00e-06 toks/s=77,685
step=70 loss=3.3159 pred=0.4324 aux=144.1765 lr=7.00e-06 toks/s=113,117
step=80 loss=nan pred=nan aux=nan lr=8.00e-06 toks/s=163,618
step=90 loss=nan pred=nan aux=nan lr=9.00e-06 toks/s=119,958
step=100 loss=nan pred=nan aux=nan lr=1.00e-05 toks/s=263,494
step=110 loss=nan pred=nan aux=nan lr=1.10e-05 toks/s=259,032
step=120 loss=nan pred=nan aux=nan lr=1.20e-05 toks/s=246,767
step=130 loss=nan pred=nan aux=nan lr=1.30e-05 toks/s=248,867
step=140 loss=nan pred=nan aux=nan lr=1.40e-05 toks/s=250,176
step=150 loss=nan pred=nan aux=nan lr=1.50e-05 toks/s=245,992
```


####  Test numeric stability

First debug will be trying to run without casting to bf16 (for sanity)
```
foundationts train
 --no-amp --no-bf16
 <same args as above>

params total=23.11M (23,108,841) active=12.49M (12,492,009)
step=10 loss=3.6148 pred=0.5895 aux=151.2670 lr=1.00e-06 toks/s=26,204
step=20 loss=3.5842 pred=0.5931 aux=149.5561 lr=2.00e-06 toks/s=126,030
step=30 loss=3.5826 pred=0.6063 aux=148.8117 lr=3.00e-06 toks/s=126,140
step=40 loss=3.4016 pred=0.5188 aux=144.1369 lr=4.00e-06 toks/s=126,006
step=50 loss=3.3548 pred=0.4704 aux=144.2234 lr=5.00e-06 toks/s=126,525
step=60 loss=3.3919 pred=0.4843 aux=145.3838 lr=6.00e-06 toks/s=125,764
step=70 loss=3.3145 pred=0.4323 aux=144.1094 lr=7.00e-06 toks/s=79,620
step=80 loss=nan pred=nan aux=nan lr=8.00e-06 toks/s=136,240
step=90 loss=nan pred=nan aux=nan lr=9.00e-06 toks/s=143,081
```


This is a huge relief. The throughput dropped quite substantially.


#### Test MOE 
Of course aux loss is incredibly high, so this is the next thing to investigate. We can try tuning this down a bit.


```
foundationts train
  --aux-loss-weight
  <same args as below>
```

Default value in above experiments was `0.2`, I tried `0.002` and `0.0002`, same issues as above.



### Data Issues
So somewhere, we're probably introudcing some carzy grads that don't get clipped.

I added a `torch.autograd.set_detect_anomaly(True)` in train and found the following stack trace.

```
step=10 loss=3.6159 pred=0.5888 aux=151.3544 lr=1.00e-06 toks/s=17,188
step=20 loss=3.5835 pred=0.5931 aux=149.5191 lr=2.00e-06 toks/s=43,301
step=30 loss=3.5840 pred=0.6058 aux=148.9071 lr=3.00e-06 toks/s=44,335
step=40 loss=3.4019 pred=0.5189 aux=144.1535 lr=4.00e-06 toks/s=44,541
step=50 loss=3.3548 pred=0.4704 aux=144.2176 lr=5.00e-06 toks/s=38,033
step=60 loss=3.3949 pred=0.4846 aux=145.5150 lr=6.00e-06 toks/s=44,241
step=70 loss=3.3160 pred=0.4324 aux=144.1805 lr=7.00e-06 toks/s=44,531
/home/ubuntu/py312/lib/python3.12/site-packages/torch/autograd/graph.py:841: UserWarning: Error detected in MulBackward0. Traceback of forward call that caused the error:
  File "/home/ubuntu/py312/bin/foundationts", line 7, in <module>
    sys.exit(main())
  File "/home/ubuntu/FoundationTS/src/foundation_ts/cli.py", line 190, in main
    train(cfg)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 495, in train
    accum_total, accum_pred, accum_aux, accum_tokens, data_iter = _train_microbatches(
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 431, in _train_microbatches
    outputs, stats = model(input_ids, attention_mask=attention_mask)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/model.py", line 118, in forward
    hidden_state, stats = dl(hidden_state, stats, attention_mask=attention_mask)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/model.py", line 83, in forward
    hidden_state, stats = exp(hidden_state, stats, attention_mask=attention_mask)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 214, in forward
    y_out = y_out + shared_expert_score * self.shared_expert(hidden_state)
 (Triggered internally at /pytorch/torch/csrc/autograd/python_anomaly_mode.cpp:122.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "/home/ubuntu/py312/bin/foundationts", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/cli.py", line 190, in main
    train(cfg)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 495, in train
    accum_total, accum_pred, accum_aux, accum_tokens, data_iter = _train_microbatches(
                                                                  ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 436, in _train_microbatches
    (total_loss / accum_steps).backward()
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_tensor.py", line 625, in backward
    torch.autograd.backward(
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
```

So the error is in this line!
```
y_out = y_out + shared_expert_score * self.shared_expert(hidden_state)
```
Which makes sense, where else would MoE trainig going wrong? This means that either the shared_expert_score or output of the shared expert goes null. My initial sense is that this happens in the shared_expert_score, and that, well softmax is softmax and does funny things. Oh and at this point, I turned off the anomaly because it's pretty slow (look at toks/s compared between the earilier runs).


To answer I added this debug block right before the line:
```
if not torch.isfinite(shared_expert_score).all():
    raise RuntimeError("shared_expert_score has NaN/Inf")
if not torch.isfinite(hidden_state).all():
    raise RuntimeError("hidden_state has NaN/Inf before shared expert")
shared_out = self.shared_expert(hidden_state)
if not torch.isfinite(shared_out).all():
    raise RuntimeError("shared_expert output has NaN/Inf")
```


And bingo! `RuntimeError: shared_expert_score has NaN/Inf`


I added one more check for hidden state at the top of the MoE Layer, it turns out that is null. So to find what introduces it, I added
is finite checks everywhere. Before and after pretty much everywhere. I wanted to check the embedding, decoder, RMSNorma, attention, each moe layer, attention (input, qkv, sdpa, out projection), the router (outputs, dispatch results, expert output, the final combine). This way, we're going to identify the first place that introduces the error.

After some determine, the large values are being introduced by the gate layer in the time embedding. To figure out why this is happneing, I modified the scaler in the data loader:

```
def max_scaler(seq: torch.Tensor) -> torch.Tensor:
    """Scale by the maximum absolute value."""
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(seq)
    seq = seq.to(torch.float32)

    # NEW!
    if not torch.isfinite(seq).all():
        raise RuntimeError("Non-finite values found before max_scaler normalization.")
    max_val = seq.abs().max()
    if max_val == 0:
        out = seq
    else:
        out = seq / max_val
    
    # NEW!
    if not torch.isfinite(out).all():
        raise RuntimeError("Non-finite values found after max_scaler normalization.")
    max_abs = out.abs().max().item()
    if max_abs > 1.0e6:
        raise RuntimeError(f"Unusually large values after max_scaler normalization: max_abs={max_abs:.2e}")
    return out
```

I also added this function that I call in the embedding layer now:
```
def _ensure_reasonable_range(name: str, tensor: torch.Tensor, max_abs: float = 1.0e6) -> None:
    max_val = tensor.abs().max().item()
    if max_val > max_abs:
        raise RuntimeError(f"Unusually large values in {name}: max_abs={max_val:.2e}")
```


So still the same thing, after the gate layer, these changes didn't help with a whole lot. So nothing crazy out of the normal was going
on so let's just change LR. I added `--learning-rate 5e-4`, still the same issue.


Now the plan is to loop over every gradient:

```

def _check_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Non-finite gradient detected in {name}")
```


That did not fix it. I tried setting a higher max grad norm and lower max grad norm, neither worked. I tried checking loss was null. It's not. 


SO it turns out, this was all just a waste of time. The original error showed us where the backward error was happneing from. Now let's actually fix it.

SO I added a check grad hook and it triggers on moe.shared_out. I think this means that somehow the grads are just exploding. To handle this, I'm going to add a RMS Norm layer right before the shared expert.

- RMS Norm before shared layer did not fix
- Clamp on the hidden state to [-10,10]
- Disable the shared expert, created a new error in the embed layer!
  - This just brought us back to the mebed_layer.output has a non-finite gradient

- Going to disable the Expert Layers now.
  - And we still get the error! So I'm adding some hooks to SDPA and the output layer to see what's up.

- Found a non-fintite gradient in the attention_out. Tried checking the attention masks to see if anything there is corrupted, everythginlooks good (just checked at least one value was unmasked)
- Now trying to clamp qkv, but that still gave the same error.
- I also quickly tried again casting to FP32 didn't fix the issue.
- Now I'm trying without the torch sdpa and a mnanual implementaiton, just so I can see for myself what it looks like.

- Wait... so the error was coming from torch sdpa this whole time... SMH!!!

- So I need to figure out how to do training with SDPA that doesn't break everything
- It always breaks on this same batch, so I finally decided to dump the batch as an output 
- AND IT IMMEIDATELY breaks, on a fresh model, so really we should have just started here. 


- And here's the issue... I wasn't using a causal mask, only a padding mask, which made training unstable

