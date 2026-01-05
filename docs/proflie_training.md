## Training profiler setup

This project supports PyTorch profiling with both TensorBoard traces and a Chrome
trace file. Enable it with `--profile`, and adjust the schedule if needed.

### Example

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
  --profile \
  --profile-dir checkpoints/profiler \
  --log-every 10 \
  --checkpoint-every 0
```

Outputs:
- TensorBoard profiler traces: `checkpoints/profiler`
- Chrome trace: `checkpoints/profiler/chrome_trace.json`

Notes:
- Profiling always collects Python stacks.
- The schedule is fixed: wait=10, warmup=10, active=1, repeat=1.
- Use `--compile` to enable `torch.compile` for steady-state performance tests.

To view the traces in TensorBoard:

```bash
tensorboard --logdir checkpoints
```

### Dataloader vs model timing

Use `--log-timers` to print and log approximate data wait vs model time per step.
This shows up in stdout and TensorBoard as:
- `train/data_time_ms`
- `train/model_time_ms`
- `train/step_time_ms`
- `train/data_time_pct`



### looks


Baseline:
```
step=160 toks/s=103,433 timing data=0.19ms model=875.52ms step=875.71ms
step=170 toks/s=86,943 timing data=0.16ms model=525.50ms step=525.66ms
step=180 toks/s=99,123 timing data=0.17ms model=528.64ms step=528.80ms
```

Torch Compile:
```
step=160 toks/s=120,287 timing data=0.09ms model=488.69ms step=488.78ms data%=0.0
step=170 toks/s=114,583 timing data=0.13ms model=551.90ms step=552.04ms data%=0.0
step=180 toks/s=117,321 timing data=0.15ms model=475.62ms step=475.76ms data%=0.0
```

Keeping compile on for the rest of these

No Amp 
```
step=160 loss=3.1208 pred=0.2401 aux=144.0378 lr=1.60e-05 toks/s=143,918
timing data=0.13ms model=450.90ms step=451.02ms data%=0.0
step=170 loss=3.0950 pred=0.2145 aux=144.0251 lr=1.70e-05 toks/s=141,380
timing data=0.13ms model=450.52ms step=450.65ms data%=0.0
step=180 loss=3.0747 pred=0.1942 aux=144.0227 lr=1.80e-05 toks/s=146,197
timing data=0.07ms model=437.36ms step=437.43ms data%=0.0
```

So ugh... this is nuts! AMP should be way faster, but turning it off, you get a perf boost. This is because I have no idea how AMP works!


Removed all the manual precision changes and rerunning with Amp
```
step=160 loss=3.1212 pred=0.2402 aux=144.0525 lr=1.60e-05 toks/s=109,124
timing data=0.14ms model=630.73ms step=630.87ms data%=0.0
step=170 loss=3.0963 pred=0.2146 aux=144.0845 lr=1.70e-05 toks/s=106,155
timing data=0.13ms model=576.33ms step=576.46ms data%=0.0
step=180 loss=3.0752 pred=0.1944 aux=144.0391 lr=1.80e-05 toks/s=108,780
timing data=0.14ms model=566.49ms step=566.63ms data%=0.0
```

Okay so that's worse...? Very confused. 

Turns out, there's a big issue that could break AMP, and that was me thinking I'm smart. I wanted to try out all of these accumulation tricks in FP32, but they actually might make my code run slower. I confirmed this by running `nvidia-smi dmon`, Memory controller utilization is high ans so is SM Utilization, so AMP should give a nice speed up. 


So I removed all of the casts and added a F.RMSNorm layer, well...

```
step=160 loss=3.1212 pred=0.2402 aux=144.0511 lr=1.60e-05 toks/s=74,084
timing data=0.14ms model=906.60ms step=906.74ms data%=0.0
step=170 loss=3.0963 pred=0.2147 aux=144.0836 lr=1.70e-05 toks/s=72,474
timing data=0.25ms model=942.18ms step=942.43ms data%=0.0
step=180 loss=3.0751 pred=0.1944 aux=144.0380 lr=1.80e-05 toks/s=76,054
timing data=0.13ms model=893.36ms step=893.50ms data%=0.0

```

Even worse!! Still at a loss


As a sanity check I stil want to see what No Amp is looking like: 
```
step=160 loss=3.1208 pred=0.2401 aux=144.0376 lr=1.60e-05 toks/s=147,639
timing data=0.09ms model=445.29ms step=445.38ms data%=0.0
step=170 loss=3.0950 pred=0.2145 aux=144.0250 lr=1.70e-05 toks/s=147,112
timing data=0.08ms model=447.62ms step=447.69ms data%=0.0
step=180 loss=3.0747 pred=0.1942 aux=144.0232 lr=1.80e-05 toks/s=147,515
timing data=0.08ms model=441.66ms step=441.74ms data%=0.0
```


Now I am noticing that I get these errors in my code when I compile and use Amp, so that's the next step to debug. When I don't use amp, I don't have the issue.

For RMSNorm that's pretty obvious and expected. For the second ne, the forward, that's in my MoE layer. It's because we get variable tokens routed to each expert and it's not really possible to precompute shapes for `torch.compile` so it's going to have to do some new things. That's the next investigation.
```
/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/utils.py:3546: UserWarning: Mismatch dtype between input and weight: input dtype = c10::BFloat16, weight dtype = float, Cannot dispatch to fused implementation. (Triggered internally at /pytorch/aten/src/ATen/native/layer_norm.cpp:344.)
  return node.target(*args, **kwargs)  # type: ignore[operator]
W0105 02:42:06.408000 48805 torch/_dynamo/convert_frame.py:1358] [9/8] torch._dynamo hit config.recompile_limit (8)
W0105 02:42:06.408000 48805 torch/_dynamo/convert_frame.py:1358] [9/8]    function: 'forward' (/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py:150)
W0105 02:42:06.408000 48805 torch/_dynamo/convert_frame.py:1358] [9/8]    last reason: 9/5: 2 <= x.size()[0]  # return F.linear(input, self.weight, self.bias)  # nn/modules/linear.py:134 in forward (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.mark_unbacked(tensor, dim))
W0105 02:42:06.408000 48805 torch/_dynamo/convert_frame.py:1358] [9/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
W0105 02:42:06.408000 48805 torch/_dynamo/convert_frame.py:1358] [9/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html
```