## Training profiler setup


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
well it didn't compile...


### MOE Layer Deep Dive
I'm just going to isolate the MOE layers until I find the one that works the best and use that. I cranked up Batch so the tflops and ms are more fun to look at.
Here's the baseline:
```
Benchmark settings batch=512 seq=4096 hidden=512 experts=8 k=2 dtype=torch.bfloat16 backward=True
MOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=15,880,819 step_ms=132.06 tflops=150.75
EfficientMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=16,034,768 step_ms=130.79 tflops=152.21
EfficientMOELayer (compiled)
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=26,188,956 step_ms=80.08 tflops=248.60
AdaptiveMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=10,492,107 step_ms=199.88 tflops=99.60
```


But interestingly, when I test with the attention mask, I see these numbers:
```
Benchmark settings batch=512 seq=4096 hidden=512 experts=8 k=2 dtype=torch.bfloat16 backward=True
MOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=15,830,824 step_ms=132.47 tflops=150.27
EfficientMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=2,301,000 step_ms=911.41 tflops=21.84
EfficientMOELayer (compiled)
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=2,399,856 step_ms=873.87 tflops=22.78
AdaptiveMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=13,139,833 step_ms=159.60 tflops=124.73
```

So the attention mask just kills performance!

I looked at some trace files for the MOE layers and found the issue. We spend 199ms in this kernel: 

```
void (anonymous namespace)::indexing_backward_kernel<c10::BFloat16, 4>(long const*, long const*, c10::BFloat16 const*, c10::BFloat16*, long, long, long, long, bool) 
```

This is caused by the attention mask altering a variable `expert_for_route`, which determines the indexing order for the expert layer input. We currently use the attetion mask like this:

```
if attention_mask is not None:
  ...
  expert_for_route = torch.where(valid_route, expert_for_route, torch.zeros_like(expert_for_route))

# do some more logic
e_idx = expert_sorted
p_idx = positions.clamp(min=0, max=C - 1)  # safe for dropped items; they'll be masked by `keep`
expert_inputs[e_idx[keep], p_idx[keep]] = x_sorted[keep]
```

This works when the attention mask is not none, because keep is deterministic, but the attention mask really messes things up with the re-indexing.


So I tried replacing with a new formulation:

```
lin = e_idx * C + p_idx  # linear slot in [E*C]
src = x_sorted * keep.to(x_sorted.dtype).unsqueeze(-1)
expert_inputs_flat = expert_inputs.view(E*C, D)
expert_inputs_flat.scatter_(0, lin.unsqueeze(-1).expand(-1, D), src)
expert_inputs = expert_inputs_flat.view(E, C, D)

expert_outputs_flat = expert_outputs.view(E * C, D)
lin_safe = torch.where(keep, lin, torch.zeros_like(lin))
```

This gets rid of a 2d indexing operations

Here's the new performance 
```
((py312) ) ubuntu@192-222-54-52:~/FoundationTS$ /home/ubuntu/py312/bin/python /home/ubuntu/FoundationTS/scripts/benchmark_moe_layers.py
Benchmark settings batch=128 seq=4096 hidden=512 experts=8 k=2 dtype=torch.bfloat16 backward=True
MOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=16,968,051 step_ms=30.90 tflops=161.07
EfficientMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=13,112,611 step_ms=39.98 tflops=124.47
EfficientMOELayer (compiled)
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=16,326,827 step_ms=32.11 tflops=154.98
```

This is better, but not really where we want to be.

It turns out there's a new monstrous backwards kernel, this one takes ~10ms. 
```
void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true, at::native::(anonymous namespace)::ReduceAdd>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long, long, at::native::(anonymous namespace)::ReduceAdd const&, c10::BFloat16) 
```

I replaced all of the index operations with a `scatter_add` and got this new perf report:

```
Benchmark settings batch=128 seq=4096 hidden=512 experts=8 k=2 dtype=torch.bfloat16 backward=True
MOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=16,942,215 step_ms=30.95 tflops=160.82
EfficientMOELayer
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=17,116,561 step_ms=30.63 tflops=162.48
EfficientMOELayer (compiled)
  params total=2.36M (2,363,904) active=791.04K (791,040)
  toks/s=20,727,827 step_ms=25.29 tflops=196.76
```




#### MoE routing examples

Below are small, self-contained snippets that mirror the routing changes we tested in
`EfficientMOELayer` for performance debugging.

Baseline gather (index_select on a flattened buffer):

```python
# expert_outputs: [E, C, D]
expert_outputs_flat = expert_outputs.view(E * C, D)
lin = e_idx * C + p_idx
lin_safe = torch.where(keep, lin, torch.zeros_like(lin))
y_sorted = expert_outputs_flat.index_select(0, lin_safe)  # [R, D]
y_sorted = y_sorted * keep.to(y_sorted.dtype).unsqueeze(-1)
y_sorted = y_sorted * gate_sorted.to(y_sorted.dtype).unsqueeze(-1)
```

Scatter-based inversion (avoids index_select in backward and is compile-friendly):

```python
# expert_outputs: [E, C, D]
expert_outputs_flat = expert_outputs.view(E * C, D)
lin_valid = lin[keep]
token_valid = token_sorted[keep]
gate_valid = gate_sorted[keep]

slot_gate = x_sorted.new_zeros((E * C,))
slot_gate = torch.scatter(slot_gate, 0, lin_valid, gate_valid.to(slot_gate.dtype))
slot_token = torch.full((E * C,), -1, device=hidden_state.device, dtype=torch.long)
slot_token = torch.scatter(slot_token, 0, lin_valid, token_valid)

slot_outputs = expert_outputs_flat * slot_gate.unsqueeze(-1)
slot_mask = slot_token >= 0

y_out = x.new_zeros((N, D))
y_out = torch.scatter_add(
    y_out,
    0,
    slot_token[slot_mask].unsqueeze(-1).expand(-1, D),
    slot_outputs[slot_mask].to(y_out.dtype),
)
```


### MOE Layer Adjustments


```
params total=538.13M (538,130,793) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=264.0115 pred=261.1106 aux=145.0443 lr=1.00e-06 toks/s=104,872 tflops=249.67 mfu=12.62% step_ms=470.78 sm_util=99.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=254.5965 pred=251.6948 aux=145.0851 lr=2.00e-06 toks/s=167,239 tflops=398.15 mfu=20.12% step_ms=389.21 sm_util=90.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=30 loss=260.2265 pred=257.3271 aux=144.9687 lr=3.00e-06 toks/s=168,132 tflops=400.27 mfu=20.23% step_ms=387.10 sm_util=99.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=252.8084 pred=249.9070 aux=145.0677 lr=4.00e-06 toks/s=167,274 tflops=398.23 mfu=20.12% step_ms=389.08 sm_util=99.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=50 loss=213.8530 pred=210.9552 aux=144.8889 lr=5.00e-06 toks/s=168,853 tflops=401.99 mfu=20.31% step_ms=385.35 sm_util=99.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=60 loss=204.3833 pred=201.4752 aux=145.4029 lr=6.00e-06 toks/s=167,954 tflops=399.85 mfu=20.20% step_ms=387.51 sm_util=94.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=70 loss=196.7672 pred=193.8608 aux=145.3186 lr=7.00e-06 toks/s=169,481 tflops=403.49 mfu=20.39% step_ms=384.09 sm_util=99.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=80 loss=204.6532 pred=201.7394 aux=145.6887 lr=8.00e-06 toks/s=168,381 tflops=400.87 mfu=20.26% step_ms=386.54 sm_util=93.0% hbm_util=54.0% mem_ctrl_util=54.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.86
```


### Another Torch.Compile (with Capacity)

So ran compile, good improvements, but....
```
device model=NVIDIA H100 80GB HBM3 precision=bf16
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] Graph break from `Tensor.item()`, consider setting:
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0]     torch._dynamo.config.capture_scalar_outputs = True
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] or:
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0]     env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] to include these operations in the captured graph.
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] 
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] Graph break: from user code at:
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0]   File "/home/ubuntu/FoundationTS/src/foundation_ts/models/tsmoe/layers.py", line 17, in torch_dynamo_resume_in__get_unpad_data_at_16
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0]     max_seqlen_in_batch = seqlens_in_batch.max().item()
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] 
W0112 02:58:28.700000 60377 torch/_dynamo/variables/tensor.py:1048] [6/0] 
step=10 loss=264.4143 pred=261.4405 aux=148.6890 lr=1.00e-06 toks/s=32,890 tflops=78.30 mfu=3.96% step_ms=1856.21 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=253.0651 pred=250.0910 aux=148.7041 lr=2.00e-06 toks/s=204,843 tflops=487.67 mfu=24.64% step_ms=317.57 sm_util=77.0% hbm_util=47.0% mem_ctrl_util=47.0%
step=30 loss=261.7725 pred=258.8101 aux=148.1219 lr=3.00e-06 toks/s=205,881 tflops=490.14 mfu=24.77% step_ms=316.02 sm_util=82.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=40 loss=242.5769 pred=239.6313 aux=147.2762 lr=4.00e-06 toks/s=205,334 tflops=488.84 mfu=24.70% step_ms=316.80 sm_util=85.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=50 loss=215.3560 pred=212.3647 aux=149.5612 lr=5.00e-06 toks/s=207,706 tflops=494.49 mfu=24.99% step_ms=313.05 sm_util=90.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=183.9454 pred=180.8614 aux=154.2012 lr=6.00e-06 toks/s=206,202 tflops=490.91 mfu=24.81% step_ms=315.49 sm_util=97.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=70 loss=206.5421 pred=203.6012 aux=147.0445 lr=7.00e-06 toks/s=208,748 tflops=496.97 mfu=25.11% step_ms=311.67 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=192.7895 pred=189.8213 aux=148.4076 lr=8.00e-06 toks/s=206,775 tflops=492.27 mfu=24.87% step_ms=314.65 sm_util=83.0% hbm_util=52.0% mem_ctrl_util=52.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.54
```

Small error with 3 options:
1. Just use `shape[-1]`, easy fix, slightly less efficient:
2. Keep the real max and allow scalar capture (set torch._dynamo.config.capture_scalar_outputs = True or env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1), or
3. Compute the max outside the compiled region and pass it in (e.g., precompute from the mask in the caller and add an optional max_seqlen arg).

Trace is `profile_traces/`.

Option #1:
```
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=265.0006 pred=262.0271 aux=148.6784 lr=1.00e-06 toks/s=61,167 tflops=145.62 mfu=7.36% step_ms=937.98 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=252.7030 pred=249.7288 aux=148.7090 lr=2.00e-06 toks/s=205,738 tflops=489.80 mfu=24.75% step_ms=316.15 sm_util=85.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=30 loss=262.0685 pred=259.1057 aux=148.1399 lr=3.00e-06 toks/s=207,131 tflops=493.12 mfu=24.92% step_ms=314.09 sm_util=91.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=40 loss=244.8419 pred=241.9019 aux=146.9976 lr=4.00e-06 toks/s=206,085 tflops=490.63 mfu=24.79% step_ms=315.62 sm_util=97.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=50 loss=220.7739 pred=217.7923 aux=149.0798 lr=5.00e-06 toks/s=208,538 tflops=496.47 mfu=25.09% step_ms=311.89 sm_util=87.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=60 loss=191.5695 pred=188.5015 aux=153.3991 lr=6.00e-06 toks/s=207,482 tflops=493.95 mfu=24.96% step_ms=313.51 sm_util=82.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=70 loss=197.8744 pred=194.8877 aux=149.3340 lr=7.00e-06 toks/s=209,586 tflops=498.96 mfu=25.21% step_ms=310.44 sm_util=90.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=80 loss=178.6148 pred=175.6185 aux=149.8148 lr=8.00e-06 toks/s=207,672 tflops=494.41 mfu=24.98% step_ms=313.18 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.54
```

Option #2:
```
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=264.9703 pred=261.9965 aux=148.6910 lr=1.00e-06 toks/s=60,184 tflops=143.28 mfu=7.24% step_ms=930.56 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=254.1951 pred=251.2219 aux=148.6612 lr=2.00e-06 toks/s=205,598 tflops=489.47 mfu=24.73% step_ms=316.37 sm_util=87.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=30 loss=261.6329 pred=258.6715 aux=148.0693 lr=3.00e-06 toks/s=206,760 tflops=492.24 mfu=24.87% step_ms=314.64 sm_util=92.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=249.1610 pred=246.2209 aux=147.0088 lr=4.00e-06 toks/s=206,375 tflops=491.32 mfu=24.83% step_ms=315.19 sm_util=98.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=50 loss=212.6327 pred=209.6446 aux=149.4023 lr=5.00e-06 toks/s=208,379 tflops=496.09 mfu=25.07% step_ms=312.07 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=60 loss=189.6552 pred=186.5679 aux=154.3697 lr=6.00e-06 toks/s=207,351 tflops=493.64 mfu=24.94% step_ms=313.71 sm_util=82.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=70 loss=178.0448 pred=175.0500 aux=149.7425 lr=7.00e-06 toks/s=209,550 tflops=498.88 mfu=25.21% step_ms=310.51 sm_util=91.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=80 loss=157.8656 pred=154.8192 aux=152.3195 lr=8.00e-06 toks/s=207,699 tflops=494.47 mfu=24.99% step_ms=313.15 sm_util=90.0% hbm_util=62.0% mem_ctrl_util=62.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.54
```

Honestly option #2 is good enough so just going to stick with that.


### Finally a batch size increase:

```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 22 \
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
  --mfu-peak-tflops 1979 \
  --compile
```

Trace file is `big_mfu_gain.json`.

So just increased to 22, that's a good size increase!
```
params total=538.13M (538,130,793) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=264.5456 pred=261.5764 aux=148.4557 lr=1.00e-06 toks/s=41,265 tflops=98.24 mfu=4.96% step_ms=2025.03 sm_util=99.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=253.4424 pred=250.4918 aux=147.5271 lr=2.00e-06 toks/s=220,477 tflops=524.89 mfu=26.52% step_ms=406.01 sm_util=85.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=30 loss=250.9666 pred=248.0139 aux=147.6335 lr=3.00e-06 toks/s=219,897 tflops=523.51 mfu=26.45% step_ms=406.99 sm_util=95.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=216.1697 pred=213.1769 aux=149.6393 lr=4.00e-06 toks/s=218,993 tflops=521.36 mfu=26.34% step_ms=408.88 sm_util=86.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=50 loss=206.1311 pred=203.1387 aux=149.6241 lr=5.00e-06 toks/s=223,780 tflops=532.75 mfu=26.92% step_ms=400.04 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=60 loss=219.9569 pred=216.9866 aux=148.5134 lr=6.00e-06 toks/s=219,638 tflops=522.89 mfu=26.42% step_ms=407.63 sm_util=90.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=198.2897 pred=195.3335 aux=147.8114 lr=7.00e-06 toks/s=219,289 tflops=522.06 mfu=26.38% step_ms=408.18 sm_util=99.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=80 loss=197.0481 pred=194.0903 aux=147.8908 lr=8.00e-06 toks/s=223,470 tflops=532.02 mfu=26.88% step_ms=400.47 sm_util=99.0% hbm_util=60.0% mem_ctrl_util=60.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=77.00
```



## Next Day
I'm making some changes to more closely match the implementation from Time-MOE. Here's what I changed:

- Attention has separate q,k,v projections instead of one qkv proj
- Added gate_proj

Attention: split qkv_proj into q_proj, k_proj, v_proj, keep o_proj in layers.py.
MoE experts: switched to gate_proj/up_proj/down_proj (SiLU), and set defaults to d_ff = 4 * hidden_size, d_expert = d_ff // k in layers.py.
Output heads: now emit horizon * input_size per head in model.py.
Embedding: input size now honors input_size (and patch_len * input_size when patched) in model.py.
Model config: added input_size to CLI/config and pass-through in config.py, cli.py, loop.py.

New Command:
```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 16 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --compile
```



```
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5663.3120 pred=5660.3594 aux=147.6366 lr=1.00e-06 toks/s=28,407 tflops=67.63 mfu=3.42% step_ms=2142.02 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=20 loss=5714.3550 pred=5711.3770 aux=148.8905 lr=2.00e-06 toks/s=199,885 tflops=475.87 mfu=24.05% step_ms=326.08 sm_util=80.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=160 loss=4943.9971 pred=4941.0352 aux=148.1076 lr=1.60e-05 toks/s=200,004 tflops=476.15 mfu=24.06% step_ms=325.82 sm_util=78.0% hbm_util=49.0% mem_ctrl_util=49.0%
step=170 loss=5163.7744 pred=5160.7422 aux=151.6148 lr=1.70e-05 toks/s=205,548 tflops=489.35 mfu=24.73% step_ms=317.03 sm_util=79.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=180 loss=4658.3608 pred=4655.3652 aux=149.7702 lr=1.80e-05 toks/s=203,467 tflops=484.39 mfu=24.48% step_ms=320.30 sm_util=96.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=190 loss=4599.5347 pred=4596.5684 aux=148.3087 lr=1.90e-05 toks/s=198,050 tflops=471.50 mfu=23.83% step_ms=329.36 sm_util=80.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=200 loss=4441.9048 pred=4438.9614 aux=147.1740 lr=2.00e-05 toks/s=203,376 tflops=484.18 mfu=24.47% step_ms=320.38 sm_util=81.0% hbm_util=53.0% mem_ctrl_util=53.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=56.28
```
I'll increase batch size a bit on the next run to up the sm_util since we aren't saturating RAM. There is a small drop in MFU, but it's pretty close.

### Sequence Packing
You can add `--pack-sequences` now to run the code!

```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 22 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --pack-sequences \
  --compile 
```

```
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5894.1631 pred=5891.1680 aux=149.7560 lr=1.00e-06 toks/s=38,768 tflops=92.30 mfu=4.66% step_ms=1948.81 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=20 loss=5763.2783 pred=5760.2578 aux=151.0268 lr=2.00e-06 toks/s=170,092 tflops=404.94 mfu=20.46% step_ms=479.55 sm_util=86.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=30 loss=5848.5884 pred=5845.5850 aux=150.1603 lr=3.00e-06 toks/s=170,122 tflops=405.01 mfu=20.47% step_ms=479.56 sm_util=98.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=5759.3818 pred=5756.3828 aux=149.9534 lr=4.00e-06 toks/s=171,299 tflops=407.81 mfu=20.61% step_ms=476.01 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=5841.9927 pred=5838.9912 aux=150.0702 lr=5.00e-06 toks/s=171,629 tflops=408.60 mfu=20.65% step_ms=475.18 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=60 loss=5788.2378 pred=5785.2373 aux=150.0143 lr=6.00e-06 toks/s=170,855 tflops=406.75 mfu=20.55% step_ms=477.29 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=70 loss=5859.9165 pred=5856.9536 aux=148.1409 lr=7.00e-06 toks/s=172,482 tflops=410.63 mfu=20.75% step_ms=472.83 sm_util=83.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=80 loss=5739.6338 pred=5736.6592 aux=148.7252 lr=8.00e-06 toks/s=169,209 tflops=402.84 mfu=20.36% step_ms=481.92 sm_util=69.0% hbm_util=39.0% mem_ctrl_util=39.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=68.62
```

Sequence packing degrades the code! I'm going to get a clean baseline without compile then with --pack sequences

```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 18 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --profile
```


Trace is `no_compile_no_pack.json`.
```
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5725.8984 pred=5722.9912 aux=145.3512 lr=1.00e-06 toks/s=100,542 tflops=239.36 mfu=12.10% step_ms=582.54 sm_util=99.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=20 loss=5739.0415 pred=5736.1396 aux=145.0949 lr=2.00e-06 toks/s=165,823 tflops=394.77 mfu=19.95% step_ms=444.01 sm_util=99.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=30 loss=5837.7710 pred=5834.8638 aux=145.3615 lr=3.00e-06 toks/s=144,374 tflops=343.71 mfu=17.37% step_ms=433.97 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0% kernels/step=612.1
step=40 loss=5752.9233 pred=5750.0132 aux=145.4969 lr=4.00e-06 toks/s=78,606 tflops=187.14 mfu=9.46% step_ms=437.07 sm_util=92.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=50 loss=5844.4873 pred=5841.5825 aux=145.2343 lr=5.00e-06 toks/s=167,670 tflops=399.17 mfu=20.17% step_ms=431.05 sm_util=93.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=5661.1875 pred=5658.2886 aux=144.9550 lr=6.00e-06 toks/s=165,818 tflops=394.76 mfu=19.95% step_ms=435.93 sm_util=95.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=5785.2524 pred=5782.3506 aux=145.0862 lr=7.00e-06 toks/s=164,636 tflops=391.95 mfu=19.81% step_ms=438.97 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=80 loss=5487.4180 pred=5484.5117 aux=145.3167 lr=8.00e-06 toks/s=168,304 tflops=400.68 mfu=20.25% step_ms=429.24 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=73.73 kernels/step=612.1
```


```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 18 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --profile \
  --compile

params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5720.7017 pred=5717.7227 aux=148.9576 lr=1.00e-06 toks/s=51,636 tflops=122.93 mfu=6.21% step_ms=1258.28 sm_util=99.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=20 loss=5754.1670 pred=5751.2090 aux=147.9119 lr=2.00e-06 toks/s=209,228 tflops=498.11 mfu=25.17% step_ms=351.80 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=30 loss=5825.7754 pred=5822.7930 aux=149.1110 lr=3.00e-06 toks/s=169,377 tflops=403.24 mfu=20.38% step_ms=346.13 sm_util=84.0% hbm_util=54.0% mem_ctrl_util=54.0% kernels/step=543.9
step=40 loss=5713.5190 pred=5710.5366 aux=149.1148 lr=4.00e-06 toks/s=114,117 tflops=271.68 mfu=13.73% step_ms=346.65 sm_util=89.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=5898.8755 pred=5895.8574 aux=150.8925 lr=5.00e-06 toks/s=212,689 tflops=506.35 mfu=25.59% step_ms=340.54 sm_util=91.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=5615.9878 pred=5612.9922 aux=149.7820 lr=6.00e-06 toks/s=209,613 tflops=499.03 mfu=25.22% step_ms=345.64 sm_util=87.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=70 loss=5751.0024 pred=5748.0010 aux=150.0754 lr=7.00e-06 toks/s=207,769 tflops=494.64 mfu=24.99% step_ms=348.77 sm_util=84.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=80 loss=5528.2930 pred=5525.2812 aux=150.5949 lr=8.00e-06 toks/s=213,671 tflops=508.69 mfu=25.70% step_ms=338.92 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=61.35 kernels/step=543.9
```



with pack sequences, no compile
```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 18   --seq-max-len 4096   --seq-stride 4096   --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979   --profile --pack-sequences

params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5927.4429 pred=5924.5342 aux=145.4337 lr=1.00e-06 toks/s=98,671 tflops=234.91 mfu=11.87% step_ms=594.84 sm_util=99.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=5675.5239 pred=5672.6094 aux=145.7256 lr=2.00e-06 toks/s=139,690 tflops=332.56 mfu=16.80% step_ms=527.21 sm_util=85.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=30 loss=5782.7520 pred=5779.8389 aux=145.6443 lr=3.00e-06 toks/s=114,699 tflops=273.07 mfu=13.80% step_ms=514.57 sm_util=79.0% hbm_util=45.0% mem_ctrl_util=45.0% kernels/step=1075.4
step=40 loss=5733.9673 pred=5731.0488 aux=145.9195 lr=4.00e-06 toks/s=96,998 tflops=230.92 mfu=11.67% step_ms=508.84 sm_util=75.0% hbm_util=42.0% mem_ctrl_util=42.0%
step=50 loss=5844.2461 pred=5841.3276 aux=145.9183 lr=5.00e-06 toks/s=140,869 tflops=335.37 mfu=16.95% step_ms=508.92 sm_util=90.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=5845.5728 pred=5842.6499 aux=146.1366 lr=6.00e-06 toks/s=140,403 tflops=334.26 mfu=16.89% step_ms=510.50 sm_util=99.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=70 loss=5818.6606 pred=5815.7349 aux=146.2972 lr=7.00e-06 toks/s=141,434 tflops=336.71 mfu=17.01% step_ms=506.66 sm_util=99.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=80 loss=5554.5957 pred=5551.6748 aux=146.0499 lr=8.00e-06 toks/s=141,443 tflops=336.73 mfu=17.02% step_ms=506.51 sm_util=99.0% hbm_util=66.0% mem_ctrl_util=66.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=73.87 kernels/step=1075.4
```


Sequence packign with compile

```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 18   --seq-max-len 4096   --seq-stride 4096   --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979   --profile --pack-sequences --compile
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5950.9756 pred=5947.9683 aux=150.3554 lr=1.00e-06 toks/s=72,723 tflops=173.13 mfu=8.75% step_ms=846.99 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=20 loss=5717.9722 pred=5714.9678 aux=150.2258 lr=2.00e-06 toks/s=168,104 tflops=400.21 mfu=20.22% step_ms=437.99 sm_util=98.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=30 loss=5782.1133 pred=5779.1030 aux=150.5211 lr=3.00e-06 toks/s=129,350 tflops=307.94 mfu=15.56% step_ms=428.51 sm_util=78.0% hbm_util=48.0% mem_ctrl_util=48.0% 
step=40 loss=5719.5879 pred=5716.5508 aux=151.8540 lr=4.00e-06 toks/s=114,992 tflops=273.76 mfu=13.83% step_ms=419.81 sm_util=73.0% hbm_util=45.0% mem_ctrl_util=45.0%
step=50 loss=5775.6450 pred=5772.6333 aux=150.5970 lr=5.00e-06 toks/s=170,538 tflops=406.00 mfu=20.52% step_ms=419.92 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=60 loss=5902.6030 pred=5899.5693 aux=151.6927 lr=6.00e-06 toks/s=169,892 tflops=404.46 mfu=20.44% step_ms=421.72 sm_util=68.0% hbm_util=39.0% mem_ctrl_util=39.0%
step=70 loss=5853.4351 pred=5850.4414 aux=149.6804 lr=7.00e-06 toks/s=171,462 tflops=408.20 mfu=20.63% step_ms=417.75 sm_util=89.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=80 loss=5595.5210 pred=5592.5396 aux=149.0769 lr=8.00e-06 toks/s=171,887 tflops=409.21 mfu=20.68% step_ms=416.65 sm_util=67.0% hbm_util=39.0% mem_ctrl_util=39.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=62.20 kernels/step=1008.6
```

So the thing is, torch comiple can't fuse as many kernels when we pack sequencs! 

We will try bucketing now to see if that improve our implementation of sequence packing.


```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 18 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --profile \
  --pack-sequences \
  --pack-buckets 4096

params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=6173.3242 pred=6170.4287 aux=144.7831 lr=1.00e-06 toks/s=22,578 tflops=53.75 mfu=2.72% step_ms=516.45 sm_util=100.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=20 loss=6181.7554 pred=6178.8623 aux=144.6484 lr=2.00e-06 toks/s=114,984 tflops=273.74 mfu=13.83% step_ms=464.27 sm_util=99.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=30 loss=6217.8257 pred=6214.9321 aux=144.6818 lr=3.00e-06 toks/s=95,345 tflops=226.99 mfu=11.47% step_ms=469.85 sm_util=99.0% hbm_util=62.0% mem_ctrl_util=62.0% kernels/step=1077.8
step=40 loss=6167.9971 pred=6165.1021 aux=144.7628 lr=4.00e-06 toks/s=64,740 tflops=154.13 mfu=7.79% step_ms=399.91 sm_util=99.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=6172.7500 pred=6169.8550 aux=144.7564 lr=5.00e-06 toks/s=106,936 tflops=254.58 mfu=12.86% step_ms=415.90 sm_util=95.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=6138.0781 pred=6135.1792 aux=144.9571 lr=6.00e-06 toks/s=102,717 tflops=244.54 mfu=12.36% step_ms=406.99 sm_util=77.0% hbm_util=48.0% mem_ctrl_util=48.0%
step=70 loss=6081.5615 pred=6078.6626 aux=144.9447 lr=7.00e-06 toks/s=120,321 tflops=286.45 mfu=14.47% step_ms=467.64 sm_util=87.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=80 loss=6096.0366 pred=6093.1382 aux=144.9331 lr=8.00e-06 toks/s=105,446 tflops=251.04 mfu=12.68% step_ms=413.37 sm_util=57.0% hbm_util=29.0% mem_ctrl_util=29.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=75.89 kernels/step=1077.8
```

That's still really bad. Let's with only one pack bucket:

```

 foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 18 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --profile \
  --pack-sequences \
  --pack-buckets 4096
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=6126.1689 pred=6123.2637 aux=145.2522 lr=1.00e-06 toks/s=42,190 tflops=100.44 mfu=5.08% step_ms=579.56 sm_util=100.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=6129.7563 pred=6126.8584 aux=144.8971 lr=2.00e-06 toks/s=135,367 tflops=322.27 mfu=16.28% step_ms=544.04 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=30 loss=6127.3545 pred=6124.4492 aux=145.2623 lr=3.00e-06 toks/s=111,514 tflops=265.48 mfu=13.41% step_ms=531.55 sm_util=95.0% hbm_util=60.0% mem_ctrl_util=60.0% kernels/step=1077.8
step=40 loss=6099.1943 pred=6096.2871 aux=145.3556 lr=4.00e-06 toks/s=94,550 tflops=225.10 mfu=11.37% step_ms=527.17 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=6089.9038 pred=6087.0034 aux=145.0211 lr=5.00e-06 toks/s=136,039 tflops=323.87 mfu=16.37% step_ms=527.30 sm_util=91.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=6072.7310 pred=6069.8267 aux=145.2080 lr=6.00e-06 toks/s=135,612 tflops=322.85 mfu=16.31% step_ms=528.91 sm_util=83.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=70 loss=6031.2783 pred=6028.3716 aux=145.3383 lr=7.00e-06 toks/s=136,081 tflops=323.97 mfu=16.37% step_ms=527.00 sm_util=79.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=80 loss=5982.1304 pred=5979.2222 aux=145.4046 lr=8.00e-06 toks/s=135,901 tflops=323.54 mfu=16.35% step_ms=527.94 sm_util=77.0% hbm_util=44.0% mem_ctrl_util=44.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=74.29 kernels/step=1077.8
```


Okay so Sequence Packing isn't worth the extra engineering effort to try to figure out why all of these kernels are being launched. 


## Trying Megablocks

```
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 80 \
  --epochs 1 \
  --batch-size 18 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 1536 \
  --log-every 10 \
  --checkpoint-every 0 \
  --log-perf-metrics \
  --mfu-peak-tflops 1979 \
  --moe-impl megablocks \
  --compile
```


## Other things to try
- I need to figure out what other MOE frameworks do for this part of the network I'm trying to optimize
