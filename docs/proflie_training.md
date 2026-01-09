## Training profiler setup

This project supports PyTorch profiling with a Chrome trace file. Enable it with
`--profile`, and adjust the schedule if needed.

### Example

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 4 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --num-expert-layers 1 \
  --hidden-size 384 \
  --d-ff 1536 \
  --d-expert 192 \
  --k 2 \
  --n-head 12 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1671
```

Outputs:
- Chrome trace: `checkpoints/profiler/chrome_trace.json`

Notes:
- Profiling always collects Python stacks.
- The schedule is fixed: wait=10, warmup=10, active=1, repeat=1.
- Use `--compile` to enable `torch.compile` for steady-state performance tests.

### Performance metrics

Use `--log-perf-metrics` to print step-level performance stats to stdout:
- `train/toks_per_sec`
- `train/step_time_ms`
- `train/tflops`
- `train/mfu` (requires `--mfu-peak-tflops`)
- `train/sm_util_pct` (requires `nvidia-ml-py`)
- `train/hbm_util_pct` (requires `nvidia-ml-py`)
- `train/mem_ctrl_util_pct` (requires `nvidia-ml-py`)
- `train/kernel_launches_per_step`
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



## Next Day

Okay I'm starting over with a new performance harness, and a new model size. My conclusion is that my model was jsut too small to see results I could reason about right now.

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 16 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --num-expert-layers 1 \
  --hidden-size 768 \
  --n-head 12 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --k 2 \
  --num-expert-layers 1 \
  --d-ff 3072 \
  --d-expert 3072 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979
```

### Baseline

```
params total=538.13M (538,130,793) active=198.39M (198,392,169)
device model=NVIDIA H100 PCIe precision=bf16
step=10 loss=3.4835 pred=0.5437 aux=146.9904 lr=1.00e-06 toks/s=41,364 tflops=98.48 mfu=5.89% step_ms=1416.51 sm_util=100.0% hbm_util=55.0%
step=20 loss=3.2919 pred=0.4054 aux=144.3282 lr=2.00e-06 toks/s=53,143 tflops=126.52 mfu=7.57% step_ms=1229.46 sm_util=100.0% hbm_util=55.0%
step=30 loss=3.2345 pred=0.3463 aux=144.4116 lr=3.00e-06 toks/s=55,486 tflops=132.10 mfu=7.91% step_ms=1177.31 sm_util=100.0% hbm_util=65.0%
step=40 loss=3.1738 pred=0.2950 aux=143.9379 lr=4.00e-06 toks/s=54,344 tflops=129.38 mfu=7.74% step_ms=1181.83 sm_util=100.0% hbm_util=59.0%
step=50 loss=3.1871 pred=0.2879 aux=144.9610 lr=5.00e-06 toks/s=53,441 tflops=127.23 mfu=7.61% step_ms=1201.98 sm_util=100.0% hbm_util=52.0%
step=60 loss=3.1827 pred=0.2985 aux=144.2100 lr=6.00e-06 toks/s=52,600 tflops=125.23 mfu=7.49% step_ms=1224.52 sm_util=100.0% hbm_util=51.0%
step=70 loss=3.1595 pred=0.2806 aux=143.9445 lr=7.00e-06 toks/s=53,434 tflops=127.21 mfu=7.61% step_ms=1201.86 sm_util=100.0% hbm_util=58.0%
step=80 loss=3.0973 pred=0.2125 aux=144.2365 lr=8.00e-06 toks/s=53,362 tflops=127.04 mfu=7.60% step_ms=1203.54 sm_util=100.0% hbm_util=62.0%
run model=NVIDIA H100 PCIe precision=bf16 peak_vram_gb=74.64
```

I want to see if I OOM with `--no-amp`
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 46.50 MiB is free. Including non-PyTorch memory, this process has 79.13 GiB memory in use. Of the allocated memory 77.91 GiB is allocated by PyTorch, and 648.28 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

Yay! SO that is working!

### Compiled
Same command as above but with `--compile`

```
params total=538.13M (538,130,793) active=198.39M (198,392,169)
device model=NVIDIA H100 PCIe precision=bf16
step=10 loss=3.6292 pred=0.5437 aux=154.2740 lr=1.00e-06 toks/s=11,664 tflops=27.77 mfu=1.66% step_ms=5467.38 sm_util=100.0% hbm_util=49.0%
step=20 loss=3.2981 pred=0.4054 aux=144.6358 lr=2.00e-06 toks/s=51,189 tflops=121.87 mfu=7.29% step_ms=1265.84 sm_util=100.0% hbm_util=47.0%
step=30 loss=3.2371 pred=0.3463 aux=144.5435 lr=3.00e-06 toks/s=53,404 tflops=127.14 mfu=7.61% step_ms=1221.13 sm_util=100.0% hbm_util=57.0%
step=40 loss=3.1869 pred=0.2950 aux=144.5970 lr=4.00e-06 toks/s=53,558 tflops=127.51 mfu=7.63% step_ms=1205.46 sm_util=100.0% hbm_util=59.0%
step=50 loss=3.2091 pred=0.2879 aux=146.0595 lr=5.00e-06 toks/s=54,534 tflops=129.83 mfu=7.77% step_ms=1178.49 sm_util=100.0% hbm_util=49.0%
step=60 loss=3.2046 pred=0.2986 aux=145.3002 lr=6.00e-06 toks/s=54,451 tflops=129.63 mfu=7.76% step_ms=1179.05 sm_util=100.0% hbm_util=49.0%
step=70 loss=3.1673 pred=0.2807 aux=144.3313 lr=7.00e-06 toks/s=49,334 tflops=117.45 mfu=7.03% step_ms=1300.57 sm_util=100.0% hbm_util=62.0%
step=80 loss=3.1043 pred=0.2126 aux=144.5864 lr=8.00e-06 toks/s=55,183 tflops=131.37 mfu=7.86% step_ms=1157.67 sm_util=100.0% hbm_util=46.0%
run model=NVIDIA H100 PCIe precision=bf16 peak_vram_gb=73.31
```

Which is basically the same, so how can I make this any better? 

I changed to `mode='max-autotune'`, and here's the key observation
```
CUDAGraph supports dynamic shapes by recording a new graph for each distinct input size. Recording too many CUDAGraphs may lead to extra overhead. We have observed 51 distinct sizes. Please consider the following options for better performance: a) padding inputs to a few fixed number of shapes; or b) set torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None to silence this warning.
```

So this is most likely a cause of having bad GEMMs in my MoE Routing Layer

Also, the process failed with the following error:
```
SingleProcess AUTOTUNE benchmarking takes 0.8018 seconds and 0.0006 seconds precompiling for 20 choices
Traceback (most recent call last):
  File "/home/ubuntu/py312/bin/foundationts", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/cli.py", line 223, in main
    train(cfg)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 565, in train
    ) = _train_microbatches(
        ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/training/loop.py", line 469, in _train_microbatches
    outputs, stats = model(input_ids, attention_mask=attention_mask)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 414, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 832, in compile_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/model.py", line 137, in forward
    hidden_state, stats = dl(hidden_state, stats, attention_mask=attention_mask)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/model.py", line 86, in forward
    hidden_state = self.attention(norm_input_state, attention_mask=attention_mask)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1875, in __call__
    result = self._torchdynamo_orig_backend(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1625, in __call__
    result = self._inner_convert(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 688, in __call__
    result = _compile(
             ^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1495, in _compile
    raise InternalTorchDynamoError(
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1434, in _compile
    guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_utils_internal.py", line 92, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1117, in compile_inner
    return _compile_inner(code, one_graph, hooks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1151, in _compile_inner
    dynamo_output = compile_frame(
                    ^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1032, in compile_frame
    bytecode, tracer_output = transform_code_object(code, transform)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/bytecode_transformation.py", line 1592, in transform_code_object
    tracer_output = transformations(instructions, code_options)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 1004, in transform
    tracer_output = trace_frame(
                    ^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 312, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 815, in trace_frame
    run_tracer()
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 797, in run_tracer
    tracer.run()
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1500, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1348, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 904, in wrapper
    return inner_fn(self, inst)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3428, in CALL
    self._call(inst)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 3422, in _call
    self.call_function(fn, args, kwargs)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1266, in call_function
    self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 212, in realize_and_forward
    return getattr(self.realize(), name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/nn_module.py", line 1010, in call_function
    return variables.UserFunctionVariable(fn, source=source).call_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 598, in call_function
    return super().call_function(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/functions.py", line 342, in call_function
    return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1288, in inline_user_function_return
    return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4129, in inline_call
    return tracer.inline_call_()
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 4332, in inline_call_
    self.run()
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1500, in run
    while self.step():
          ^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 1348, in step
    self.dispatch_table[inst.opcode](self, inst)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2450, in LOAD_ATTR
    self._load_attr(inst)
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 2438, in _load_attr
    result = BuiltinVariable(getattr).call_function(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builtin.py", line 1347, in call_function
    return handler(tx, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builtin.py", line 1154, in builtin_dispatch
    rv = fn(tx, args, kwargs)
         ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builtin.py", line 1032, in call_self_handler
    result = self_handler(tx, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builtin.py", line 2338, in call_getattr
    return obj.var_getattr(tx, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/nn_module.py", line 1177, in var_getattr
    return super().var_getattr(tx, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/user_defined.py", line 1407, in var_getattr
    if isinstance(
       ^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/base.py", line 227, in __instancecheck__
    instance = instance.realize()
               ^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 72, in realize
    self._cache.realize()
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/lazy.py", line 33, in realize
    self.vt = builder.VariableBuilder(tx, self.source)(self.value)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 446, in __call__
    vt = self._wrap(value)
         ^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 671, in _wrap
    return type_dispatch(self, value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 2117, in wrap_tensor
    example_value = wrap_to_fake_tensor_and_record(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 3507, in wrap_to_fake_tensor_and_record
    fake_e = wrap_fake_exception(
             ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/utils.py", line 2864, in wrap_fake_exception
    return fn()
           ^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 3508, in <lambda>
    lambda: tx.fake_mode.from_tensor(
            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 2993, in from_tensor
    return self.fake_tensor_converter.from_real_tensor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 404, in from_real_tensor
    out = self.meta_converter(
          ^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_subclasses/meta_utils.py", line 1898, in __call__
    t_desc = self.describer.describe_tensor(t, trace=trace)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/py312/lib/python3.12/site-packages/torch/_subclasses/meta_utils.py", line 310, in describe_tensor
    storage = self.describe_storage(t.untyped_storage(), trace=trace)
                                    ^^^^^^^^^^^^^^^^^^^
torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 119, in forward
    cos, sin = self.rotary_emb(q, seq_len=T)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 84, in forward
    self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 71, in _set_cos_sin_cache
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

from user code:
   File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 119, in forward
    cos, sin = self.rotary_emb(q, seq_len=T)
  File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 81, in forward
    or self.cos_cached.device != x.device

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```


I went back and ran the baselin again but add the memory controller to the output:

```
device model=NVIDIA H100 PCIe precision=bf16
step=10 loss=3.4835 pred=0.5437 aux=146.9905 lr=1.00e-06 toks/s=39,256 tflops=93.46 mfu=5.59% step_ms=1445.95 sm_util=90.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=20 loss=3.2919 pred=0.4054 aux=144.3283 lr=2.00e-06 toks/s=52,180 tflops=124.23 mfu=7.43% step_ms=1250.99 sm_util=100.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=30 loss=3.2345 pred=0.3463 aux=144.4121 lr=3.00e-06 toks/s=53,647 tflops=127.72 mfu=7.64% step_ms=1214.25 sm_util=93.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=40 loss=3.1738 pred=0.2950 aux=143.9383 lr=4.00e-06 toks/s=53,768 tflops=128.01 mfu=7.66% step_ms=1203.65 sm_util=100.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=50 loss=3.1872 pred=0.2879 aux=144.9655 lr=5.00e-06 toks/s=51,840 tflops=123.42 mfu=7.39% step_ms=1239.80 sm_util=100.0% hbm_util=48.0% mem_ctrl_util=48.0%
step=60 loss=3.1827 pred=0.2985 aux=144.2108 lr=6.00e-06 toks/s=52,775 tflops=125.64 mfu=7.52% step_ms=1220.32 sm_util=100.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=70 loss=3.1594 pred=0.2806 aux=143.9382 lr=7.00e-06 toks/s=53,154 tflops=126.54 mfu=7.57% step_ms=1208.47 sm_util=90.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=80 loss=3.0965 pred=0.2127 aux=144.1904 lr=8.00e-06 toks/s=53,296 tflops=126.88 mfu=7.59% step_ms=1210.48 sm_util=100.0% hbm_util=57.0% mem_ctrl_util=57.0%
run model=NVIDIA H100 PCIe precision=bf16 peak_vram_gb=73.78
```

So this means the overhead for debugging the job is from a bunch of tiny kernel launches. This is kind of expected from MOE. My focus right now is to clean up the routing and expert layer optimizations.


Ohhh so here's a big win, I was using a dynamic mask and got the following error:
```
/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py:139: UserWarning: Flash Attention does not support non-null attn_mask. (Triggered internally at /pytorch/aten/src/ATen/native/transformers/sdp_utils_cpp.h:262.)
```

I caught this because 1) I was looking at my trace file and was suspicious of the kernel

```
fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::Params) 
```
This obviously doesn't look like FA, so I want to test, I enforced using:

```
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=combined_mask, is_causal=False
    )
```


#### Flash Attention

```
params total=538.13M (538,130,793) active=198.39M (198,392,169)
device model=NVIDIA H100 PCIe precision=bf16
step=10 loss=3.4723 pred=0.5438 aux=146.4258 lr=1.00e-06 toks/s=49,275 tflops=117.31 mfu=7.02% step_ms=1163.91 sm_util=98.0% hbm_util=86.0% mem_ctrl_util=86.0%
step=20 loss=3.2807 pred=0.3980 aux=144.1323 lr=2.00e-06 toks/s=78,262 tflops=186.32 mfu=11.15% step_ms=833.95 sm_util=100.0% hbm_util=87.0% mem_ctrl_util=87.0%
step=30 loss=3.2408 pred=0.3415 aux=144.9621 lr=3.00e-06 toks/s=83,713 tflops=199.30 mfu=11.93% step_ms=779.90 sm_util=100.0% hbm_util=84.0% mem_ctrl_util=84.0%
step=40 loss=3.1732 pred=0.2926 aux=144.0286 lr=4.00e-06 toks/s=84,786 tflops=201.85 mfu=12.08% step_ms=764.21 sm_util=100.0% hbm_util=87.0% mem_ctrl_util=87.0%
step=50 loss=3.1755 pred=0.2862 aux=144.4656 lr=5.00e-06 toks/s=83,679 tflops=199.21 mfu=11.92% step_ms=772.23 sm_util=92.0% hbm_util=82.0% mem_ctrl_util=82.0%
step=60 loss=3.1828 pred=0.2963 aux=144.3224 lr=6.00e-06 toks/s=82,145 tflops=195.56 mfu=11.70% step_ms=788.43 sm_util=100.0% hbm_util=90.0% mem_ctrl_util=90.0%
step=70 loss=3.1560 pred=0.2819 aux=143.7047 lr=7.00e-06 toks/s=87,598 tflops=208.54 mfu=12.48% step_ms=739.79 sm_util=100.0% hbm_util=85.0% mem_ctrl_util=85.0%
step=80 loss=3.1001 pred=0.2135 aux=144.3286 lr=8.00e-06 toks/s=89,583 tflops=213.27 mfu=12.76% step_ms=719.97 sm_util=100.0% hbm_util=84.0% mem_ctrl_util=84.0%
run model=NVIDIA H100 PCIe precision=bf16 peak_vram_gb=59.66
```


## Next Day, new baseline
I actually spent some time thinking about the spec sheet and realized that I was not using the right mfu peak flops. I decided to swap to the SXM instance from lambda to test it out. It gave a new baseline for the same config setup. I'm noticing that Peak vram isn't maxed out. I'm going to keep it fixed right now because I don't want to venture to far from powers of 2 and I'm at 1024 right now, with definitely OOM.

```
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4723 pred=0.5438 aux=146.4259 lr=1.00e-06 toks/s=77,863 tflops=185.37 mfu=9.37% step_ms=686.02 sm_util=100.0% hbm_util=75.0% mem_ctrl_util=75.0%
step=20 loss=3.2807 pred=0.3980 aux=144.1319 lr=2.00e-06 toks/s=126,711 tflops=301.66 mfu=15.24% step_ms=511.60 sm_util=100.0% hbm_util=78.0% mem_ctrl_util=78.0%
step=30 loss=3.2408 pred=0.3415 aux=144.9618 lr=3.00e-06 toks/s=128,574 tflops=306.10 mfu=15.47% step_ms=500.61 sm_util=84.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=3.1732 pred=0.2926 aux=144.0277 lr=4.00e-06 toks/s=131,520 tflops=313.11 mfu=15.82% step_ms=489.07 sm_util=92.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=50 loss=3.1755 pred=0.2862 aux=144.4636 lr=5.00e-06 toks/s=131,273 tflops=312.52 mfu=15.79% step_ms=490.27 sm_util=94.0% hbm_util=75.0% mem_ctrl_util=75.0%
step=60 loss=3.1826 pred=0.2963 aux=144.3156 lr=6.00e-06 toks/s=132,127 tflops=314.56 mfu=15.89% step_ms=487.00 sm_util=100.0% hbm_util=81.0% mem_ctrl_util=81.0%
step=70 loss=3.1557 pred=0.2819 aux=143.6907 lr=7.00e-06 toks/s=133,710 tflops=318.32 mfu=16.09% step_ms=481.74 sm_util=75.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=80 loss=3.1004 pred=0.2135 aux=144.3428 lr=8.00e-06 toks/s=134,339 tflops=319.82 mfu=16.16% step_ms=478.67 sm_util=100.0% hbm_util=76.0% mem_ctrl_util=76.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.75
```


### Fixing Torch Compile
So to fix torch compile, I need to make the experts not have dynamic shape. Right now, I compute experts doing the sort approach.
```
y_sorted = torch.empty_like(x_sorted)

for i, exp in enumerate(self.expert_layers):
    s_i, t = starts[i], offsets[i]
    if s_i == t:
        continue

    y_sorted[s_i:t] = exp(x_sorted[s_i:t])
```

This actually isn't great for compute because you have to launch separate kernels which means more overhead to move data/launch kernels. SO instead I'm going to make one large batched expert and add capactiy. These were the results:


#### Capacity
```
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=260.1877 pred=257.2556 aux=146.6084 lr=1.00e-06 toks/s=97,833 tflops=232.91 mfu=11.77% step_ms=525.89 sm_util=99.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=20 loss=257.3439 pred=254.4228 aux=146.0539 lr=2.00e-06 toks/s=143,708 tflops=342.13 mfu=17.29% step_ms=453.71 sm_util=99.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=30 loss=263.2367 pred=260.2896 aux=147.3550 lr=3.00e-06 toks/s=148,691 tflops=353.99 mfu=17.89% step_ms=438.56 sm_util=99.0% hbm_util=77.0% mem_ctrl_util=77.0%
step=40 loss=249.2698 pred=246.3304 aux=146.9663 lr=4.00e-06 toks/s=150,914 tflops=359.28 mfu=18.15% step_ms=431.99 sm_util=95.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=50 loss=246.9605 pred=244.0118 aux=147.4350 lr=5.00e-06 toks/s=148,779 tflops=354.20 mfu=17.90% step_ms=438.29 sm_util=96.0% hbm_util=73.0% mem_ctrl_util=73.0%
step=60 loss=218.4926 pred=215.5468 aux=147.2906 lr=6.00e-06 toks/s=148,668 tflops=353.93 mfu=17.88% step_ms=438.64 sm_util=89.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=70 loss=203.1577 pred=200.2224 aux=146.7654 lr=7.00e-06 toks/s=154,296 tflops=367.33 mfu=18.56% step_ms=422.57 sm_util=99.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=80 loss=220.8421 pred=217.9195 aux=146.1325 lr=8.00e-06 toks/s=147,688 tflops=351.60 mfu=17.77% step_ms=441.54 sm_util=99.0% hbm_util=76.0% mem_ctrl_util=76.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=78.14
```


#### Capactiy Compiled
well it didn't compile
```

```



#### TODO
- I need to reason about why the two MOE layers have different performance and why the "Efficient" one is much worse (half perf)
- I need to figure out what other MOE frameworks do for this part of the network I'm trying to optimize
