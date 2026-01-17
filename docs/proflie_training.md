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

- Attention: split qkv_proj into q_proj, k_proj, v_proj, keep o_proj in layers.py.
- MoE experts: switched to gate_proj/up_proj/down_proj (SiLU), and set defaults to d_ff = 4 * hidden_size, d_expert = d_ff // k in layers.py.
- Output heads: now emit horizon * input_size per head in model.py.
- Embedding: input size now honors input_size (and patch_len * input_size when patched) in model.py.
- Model config: added input_size to CLI/config and pass-through in config.py, cli.py, loop.py.

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


## Next Day
There's a bug with how I'm calculating capacity. Here is the diff:


```
         e_idx = expert_sorted
         p_idx = positions.clamp(min=0, max=C - 1)
         lin = e_idx * C + p_idx  # linear slot in [E*C]
-        src = x_sorted * keep.to(x_sorted.dtype).unsqueeze(-1)
+
+        lin_valid = lin[keep]
+        src_valid = x_sorted[keep]
+
         expert_inputs_flat = x_sorted.new_zeros((E * C, D))
-        expert_inputs_flat = torch.scatter(
-            expert_inputs_flat, 0, lin.unsqueeze(-1).expand(-1, D), src
-        )
+        expert_inputs_flat.scatter_(0, lin_valid[:, None].expand(-1, D), src_valid)
         expert_inputs = expert_inputs_flat.view(E, C, D)
 
         # Run experts and gather outputs back per route.
```


```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 22   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979   --compile
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5714.7065 pred=5711.7275 aux=148.9415 lr=1.00e-06 toks/s=39,978 tflops=95.18 mfu=4.81% step_ms=2105.83 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=5757.9424 pred=5754.9678 aux=148.7356 lr=2.00e-06 toks/s=219,988 tflops=523.73 mfu=26.46% step_ms=407.89 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=30 loss=5809.2651 pred=5806.2856 aux=148.9866 lr=3.00e-06 toks/s=222,178 tflops=528.94 mfu=26.73% step_ms=403.79 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=40 loss=5754.1855 pred=5751.2173 aux=148.4094 lr=4.00e-06 toks/s=223,266 tflops=531.53 mfu=26.86% step_ms=401.75 sm_util=97.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=50 loss=5711.2402 pred=5708.2461 aux=149.7059 lr=5.00e-06 toks/s=221,638 tflops=527.65 mfu=26.66% step_ms=404.82 sm_util=93.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=60 loss=5781.4434 pred=5778.4858 aux=147.8750 lr=6.00e-06 toks/s=219,441 tflops=522.42 mfu=26.40% step_ms=408.94 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=70 loss=5613.3340 pred=5610.3887 aux=147.2765 lr=7.00e-06 toks/s=226,965 tflops=540.34 mfu=27.30% step_ms=395.24 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=5644.0732 pred=5641.1084 aux=148.2337 lr=8.00e-06 toks/s=222,678 tflops=530.13 mfu=26.79% step_ms=402.93 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=74.16
```


### Trying Grouped GEMM over Batched MM
I'm going to removing the BMM with a single matmul, since we have fixed capacity now.


```
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5862.3252 pred=5859.3149 aux=150.5065 lr=1.00e-06 toks/s=47,730 tflops=113.63 mfu=5.74% step_ms=1749.83 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=5787.9414 pred=5784.9849 aux=147.8340 lr=2.00e-06 toks/s=152,709 tflops=363.55 mfu=18.37% step_ms=588.38 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=30 loss=5791.8892 pred=5788.9092 aux=149.0090 lr=3.00e-06 toks/s=153,615 tflops=365.71 mfu=18.48% step_ms=584.99 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=40 loss=5816.6387 pred=5813.7070 aux=146.5937 lr=4.00e-06 toks/s=150,134 tflops=357.42 mfu=18.06% step_ms=598.70 sm_util=99.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=50 loss=5774.9844 pred=5771.9956 aux=149.4340 lr=5.00e-06 toks/s=152,662 tflops=363.44 mfu=18.37% step_ms=588.54 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=60 loss=5870.4336 pred=5867.4805 aux=147.6672 lr=6.00e-06 toks/s=153,347 tflops=365.07 mfu=18.45% step_ms=586.04 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=70 loss=5766.1392 pred=5763.1636 aux=148.7864 lr=7.00e-06 toks/s=152,464 tflops=362.97 mfu=18.34% step_ms=589.32 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=80 loss=5668.5630 pred=5665.5996 aux=148.1573 lr=8.00e-06 toks/s=155,015 tflops=369.05 mfu=18.65% step_ms=579.67 sm_util=99.0% hbm_util=64.0% mem_ctrl_util=64.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=77.29
```

That's a big drop on MFU, but `sm_util` stays really high, I really want to know how many kernels this launches.

```
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5838.4888 pred=5835.4766 aux=150.6215 lr=1.00e-06 toks/s=65,393 tflops=155.68 mfu=7.87% step_ms=1244.45 sm_util=99.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=5778.5908 pred=5775.6284 aux=148.1096 lr=2.00e-06 toks/s=155,125 tflops=369.31 mfu=18.66% step_ms=580.34 sm_util=99.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=30 loss=5795.1992 pred=5792.2061 aux=149.6616 lr=3.00e-06 toks/s=133,432 tflops=317.66 mfu=16.05% step_ms=576.60 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0% kernels/step=900.6
step=40 loss=5840.8833 pred=5837.9482 aux=146.7554 lr=4.00e-06 toks/s=86,088 tflops=204.95 mfu=10.36% step_ms=583.31 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=50 loss=5768.9014 pred=5765.9111 aux=149.5125 lr=5.00e-06 toks/s=154,632 tflops=368.13 mfu=18.60% step_ms=574.92 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=60 loss=5884.1382 pred=5881.1924 aux=147.3007 lr=6.00e-06 toks/s=154,449 tflops=367.70 mfu=18.58% step_ms=575.64 sm_util=99.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=70 loss=5774.7158 pred=5771.7852 aux=146.5394 lr=7.00e-06 toks/s=152,636 tflops=363.38 mfu=18.36% step_ms=582.30 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=80 loss=5670.2783 pred=5667.3462 aux=146.6033 lr=8.00e-06 toks/s=155,632 tflops=370.51 mfu=18.72% step_ms=570.84 sm_util=99.0% hbm_util=64.0% mem_ctrl_util=64.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=77.29 kernels/step=900.6
```
Yikes, that's pretty bad. 

The “fused op” idea is a grouped 1x1 Conv1d implementation: we reshape [E,C,H] into [1, E*H, C] and used groups=E so a single conv call performs all experts’ linear projections with fixed shapes. That can work because grouped 1x1 conv is equivalent to per‑expert matmul, and in some backends it lowers to a single larger GEMM or more compiler‑friendly kernel. It might have helped by reducing dynamic shapes and making compilation/kernel selection simpler — but in this run it ended up launching many more kernels.


### Token Rounding


```
((py312) ) ubuntu@192-222-55-188:~/FoundationTS$ foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 22   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979   --compile --moe-m-tile 128
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5685.3545 pred=5682.1230 aux=161.5728 lr=1.00e-06 toks/s=55,049 tflops=131.05 mfu=6.62% step_ms=1473.55 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=20 loss=5704.7617 pred=5701.5566 aux=160.2583 lr=2.00e-06 toks/s=220,479 tflops=524.89 mfu=26.52% step_ms=407.23 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=30 loss=5812.3931 pred=5809.2720 aux=156.0482 lr=3.00e-06 toks/s=221,802 tflops=528.05 mfu=26.68% step_ms=404.51 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=40 loss=5768.9258 pred=5765.8037 aux=156.1004 lr=4.00e-06 toks/s=223,781 tflops=532.76 mfu=26.92% step_ms=400.82 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=50 loss=5728.0068 pred=5724.8354 aux=158.5806 lr=5.00e-06 toks/s=221,416 tflops=527.13 mfu=26.64% step_ms=405.22 sm_util=90.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=60 loss=5714.3804 pred=5711.2510 aux=156.4693 lr=6.00e-06 toks/s=219,507 tflops=522.58 mfu=26.41% step_ms=408.69 sm_util=87.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=70 loss=5561.7061 pred=5558.4844 aux=161.0790 lr=7.00e-06 toks/s=227,722 tflops=542.14 mfu=27.39% step_ms=393.92 sm_util=92.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=80 loss=5648.5649 pred=5645.4238 aux=157.0536 lr=8.00e-06 toks/s=223,646 tflops=532.44 mfu=26.90% step_ms=401.20 sm_util=90.0% hbm_util=59.0% mem_ctrl_util=59.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=74.30
```

It doesn't make a difference. I want to see what kind of difference capacity factor makes:
```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 22   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979 --moe-m-tile 128 --compile --capacity-factor 1.5

params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5921.3550 pred=5918.3101 aux=152.2528 lr=1.00e-06 toks/s=46,723 tflops=111.23 mfu=5.62% step_ms=1592.75 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=5694.6294 pred=5691.4116 aux=160.8899 lr=2.00e-06 toks/s=206,130 tflops=490.73 mfu=24.80% step_ms=395.69 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=30 loss=5845.6533 pred=5842.5303 aux=156.1550 lr=3.00e-06 toks/s=211,428 tflops=503.35 mfu=25.43% step_ms=385.97 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=5700.0518 pred=5696.8218 aux=161.4991 lr=4.00e-06 toks/s=205,221 tflops=488.57 mfu=24.69% step_ms=397.53 sm_util=88.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=50 loss=5700.0317 pred=5696.8970 aux=156.7410 lr=5.00e-06 toks/s=208,861 tflops=497.24 mfu=25.13% step_ms=390.44 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=60 loss=5579.6064 pred=5576.4004 aux=160.2964 lr=6.00e-06 toks/s=207,413 tflops=493.79 mfu=24.95% step_ms=393.19 sm_util=85.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=70 loss=5544.9438 pred=5541.7334 aux=160.5317 lr=7.00e-06 toks/s=207,518 tflops=494.04 mfu=24.96% step_ms=393.04 sm_util=97.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=80 loss=5619.4175 pred=5616.3027 aux=155.7427 lr=8.00e-06 toks/s=208,892 tflops=497.31 mfu=25.13% step_ms=390.60 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=74.05
```



```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 22   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 1979 --moe-m-tile 128 --compile --capacity-factor 1.1
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5704.0186 pred=5700.7891 aux=161.4736 lr=1.00e-06 toks/s=58,163 tflops=138.47 mfu=7.00% step_ms=1392.21 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=5941.6650 pred=5938.4463 aux=160.9367 lr=2.00e-06 toks/s=227,899 tflops=542.56 mfu=27.42% step_ms=393.68 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=30 loss=5816.5107 pred=5813.3936 aux=155.8695 lr=3.00e-06 toks/s=230,047 tflops=547.67 mfu=27.67% step_ms=389.95 sm_util=87.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=40 loss=5722.0522 pred=5718.9180 aux=156.7047 lr=4.00e-06 toks/s=230,510 tflops=548.78 mfu=27.73% step_ms=389.05 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=50 loss=5713.4688 pred=5710.3086 aux=158.0142 lr=5.00e-06 toks/s=228,207 tflops=543.29 mfu=27.45% step_ms=393.08 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=60 loss=5613.9873 pred=5610.7930 aux=159.7135 lr=6.00e-06 toks/s=179,653 tflops=427.70 mfu=21.61% step_ms=499.79 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=5533.1533 pred=5529.9326 aux=161.0394 lr=7.00e-06 toks/s=235,687 tflops=561.10 mfu=28.35% step_ms=380.55 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=80 loss=5538.1479 pred=5534.9746 aux=158.6701 lr=8.00e-06 toks/s=231,288 tflops=550.63 mfu=27.82% step_ms=387.69 sm_util=88.0% hbm_util=58.0% mem_ctrl_util=58.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=71.04
```


```
 foundationts train   --dataset-path time300b_selected   --steps-per-epoch 80   --epochs 1   --batch-size 24   --seq-max-len 4096   --seq-strid
e 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-p
eak-tflops 1979 --moe-m-tile 128 --compile --capacity-factor 0.9
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5794.5264 pred=5791.4209 aux=155.2703 lr=1.00e-06 toks/s=42,325 tflops=100.76 mfu=5.09% step_ms=2161.24 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=5675.9614 pred=5672.7734 aux=159.3875 lr=2.00e-06 toks/s=239,843 tflops=570.99 mfu=28.85% step_ms=408.05 sm_util=90.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=30 loss=5789.0942 pred=5785.9863 aux=155.3979 lr=3.00e-06 toks/s=240,802 tflops=573.28 mfu=28.97% step_ms=406.39 sm_util=99.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=5646.1543 pred=5642.8652 aux=164.4464 lr=4.00e-06 toks/s=245,888 tflops=585.39 mfu=29.58% step_ms=397.94 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=5622.1694 pred=5618.9629 aux=160.3261 lr=5.00e-06 toks/s=240,137 tflops=571.70 mfu=28.89% step_ms=407.42 sm_util=90.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=60 loss=5537.4404 pred=5534.1572 aux=164.1706 lr=6.00e-06 toks/s=242,876 tflops=578.22 mfu=29.22% step_ms=402.90 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=5624.0742 pred=5620.9307 aux=157.1815 lr=7.00e-06 toks/s=243,383 tflops=579.42 mfu=29.28% step_ms=402.00 sm_util=86.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=80 loss=5461.6753 pred=5458.3989 aux=163.8270 lr=8.00e-06 toks/s=243,059 tflops=578.65 mfu=29.24% step_ms=402.54 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=72.91
```



## Lmao, changing MFU calculation

I was using the wrong theoretical tflops, because I don't exploit sparsity. My actual max tflops are 989, and I added a function estimate how many flops my model is doing per pass.

```
foundationts train   --dataset-path time300b_selected   --steps-per-epoch 200   --epochs 1   --batch-size 24   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-experts 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 989 --moe-m-tile 64 --compile --capacity-factor 0.9
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5810.2383 pred=5807.1323 aux=155.2937 lr=1.00e-06 toks/s=80,250 tflops=127.77 mfu=12.92% step_ms=1057.48 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=5718.8608 pred=5715.6758 aux=159.2422 lr=2.00e-06 toks/s=237,581 tflops=378.26 mfu=38.25% step_ms=412.04 sm_util=88.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=30 loss=5778.2363 pred=5775.1421 aux=154.7197 lr=3.00e-06 toks/s=238,892 tflops=380.34 mfu=38.46% step_ms=409.76 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=5646.9473 pred=5643.6606 aux=164.3323 lr=4.00e-06 toks/s=244,518 tflops=389.30 mfu=39.36% step_ms=400.27 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=50 loss=5631.3750 pred=5628.1675 aux=160.3664 lr=5.00e-06 toks/s=238,203 tflops=379.25 mfu=38.35% step_ms=410.85 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=60 loss=5596.4663 pred=5593.1494 aux=165.8363 lr=6.00e-06 toks/s=240,030 tflops=382.15 mfu=38.64% step_ms=407.83 sm_util=91.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=70 loss=5633.7559 pred=5630.6226 aux=156.6655 lr=7.00e-06 toks/s=243,368 tflops=387.47 mfu=39.18% step_ms=402.23 sm_util=89.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=5488.2505 pred=5484.9912 aux=162.9748 lr=8.00e-06 toks/s=242,778 tflops=386.53 mfu=39.08% step_ms=403.14 sm_util=97.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=90 loss=5628.0044 pred=5624.9395 aux=153.2491 lr=9.00e-06 toks/s=241,032 tflops=383.75 mfu=38.80% step_ms=406.00 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=100 loss=5419.8745 pred=5416.7046 aux=158.4884 lr=1.00e-05 toks/s=245,035 tflops=390.12 mfu=39.45% step_ms=399.37 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=110 loss=5267.5776 pred=5264.3809 aux=159.8483 lr=1.10e-05 toks/s=241,890 tflops=385.12 mfu=38.94% step_ms=404.50 sm_util=87.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=120 loss=5112.1431 pred=5108.9219 aux=161.0677 lr=1.20e-05 toks/s=247,385 tflops=393.87 mfu=39.82% step_ms=395.47 sm_util=89.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=130 loss=5209.6396 pred=5206.4927 aux=157.3450 lr=1.30e-05 toks/s=237,649 tflops=378.36 mfu=38.26% step_ms=411.89 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=140 loss=5085.7231 pred=5082.4634 aux=162.9875 lr=1.40e-05 toks/s=244,175 tflops=388.75 mfu=39.31% step_ms=400.81 sm_util=93.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=150 loss=5061.4946 pred=5058.3101 aux=159.2247 lr=1.50e-05 toks/s=241,303 tflops=384.18 mfu=38.85% step_ms=405.63 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=160 loss=4861.8677 pred=4858.6875 aux=159.0014 lr=1.60e-05 toks/s=244,385 tflops=389.09 mfu=39.34% step_ms=400.47 sm_util=84.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=170 loss=4824.0610 pred=4820.9551 aux=155.3020 lr=1.70e-05 toks/s=242,515 tflops=386.11 mfu=39.04% step_ms=403.68 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=180 loss=4761.2358 pred=4758.1802 aux=152.7898 lr=1.80e-05 toks/s=238,334 tflops=379.45 mfu=38.37% step_ms=410.69 sm_util=90.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=190 loss=4424.8398 pred=4421.6089 aux=161.5480 lr=1.90e-05 toks/s=242,044 tflops=385.36 mfu=38.96% step_ms=404.14 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=200 loss=4447.7373 pred=4444.5767 aux=158.0322 lr=2.00e-05 toks/s=243,534 tflops=387.73 mfu=39.20% step_ms=401.73 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=72.45
```



## Correctness check:

```
((py312) ) ubuntu@192-222-55-188:~/FoundationTS$ foundationts train   --dataset-path time300b_selected   --steps-per-epoch 1000   --epo
chs 1   --batch-size 22   --seq-max-len 4096   --seq-stride 4096 --hidden-size 768   --n-decoder-layers 12   --n-head 12   --num-expert
s 8   --k 2   --d-ff 3072   --d-expert 1536   --log-every 10   --checkpoint-every 0   --log-perf-metrics   --mfu-peak-tflops 989 --moe-
m-tile 64 --compile --capacity-factor 1.25
params total=453.20M (453,196,137) active=198.39M (198,392,169)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=5699.4336 pred=5696.2012 aux=161.6187 lr=1.00e-06 toks/s=73,422 tflops=129.99 mfu=13.14% step_ms=1061.18 sm_util=98.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=20 loss=5784.9434 pred=5781.7437 aux=159.9814 lr=2.00e-06 toks/s=220,900 tflops=391.09 mfu=39.54% step_ms=406.21 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=30 loss=5823.4453 pred=5820.3174 aux=156.3893 lr=3.00e-06 toks/s=223,067 tflops=394.93 mfu=39.93% step_ms=402.21 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=5816.1714 pred=5813.0439 aux=156.3684 lr=4.00e-06 toks/s=223,921 tflops=396.44 mfu=40.09% step_ms=400.59 sm_util=88.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=50 loss=5720.4209 pred=5717.2344 aux=159.3245 lr=5.00e-06 toks/s=222,360 tflops=393.68 mfu=39.81% step_ms=403.49 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=60 loss=5704.9536 pred=5701.7988 aux=157.7492 lr=6.00e-06 toks/s=220,035 tflops=389.56 mfu=39.39% step_ms=407.72 sm_util=93.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=70 loss=5570.0225 pred=5566.8140 aux=160.4364 lr=7.00e-06 toks/s=228,194 tflops=404.01 mfu=40.85% step_ms=393.11 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=80 loss=5634.0210 pred=5630.8721 aux=157.4379 lr=8.00e-06 toks/s=223,562 tflops=395.81 mfu=40.02% step_ms=401.31 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=90 loss=5666.5781 pred=5663.4775 aux=155.0228 lr=9.00e-06 toks/s=222,034 tflops=393.10 mfu=39.75% step_ms=404.04 sm_util=92.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=100 loss=5446.0269 pred=5442.7588 aux=163.4049 lr=1.00e-05 toks/s=223,164 tflops=395.10 mfu=39.95% step_ms=401.94 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=110 loss=5404.6831 pred=5401.5254 aux=157.8755 lr=1.10e-05 toks/s=225,546 tflops=399.32 mfu=40.38% step_ms=397.81 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=120 loss=5574.6748 pred=5571.6523 aux=151.1198 lr=1.20e-05 toks/s=222,147 tflops=393.30 mfu=39.77% step_ms=403.80 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=130 loss=5405.9023 pred=5402.8462 aux=152.8198 lr=1.30e-05 toks/s=226,481 tflops=400.98 mfu=40.54% step_ms=396.06 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=140 loss=5182.5557 pred=5179.4243 aux=156.5585 lr=1.40e-05 toks/s=219,239 tflops=388.15 mfu=39.25% step_ms=409.18 sm_util=92.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=150 loss=5327.5771 pred=5324.5625 aux=150.7379 lr=1.50e-05 toks/s=222,397 tflops=393.74 mfu=39.81% step_ms=403.36 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=160 loss=5321.3237 pred=5318.3223 aux=150.0620 lr=1.60e-05 toks/s=222,149 tflops=393.31 mfu=39.77% step_ms=403.88 sm_util=97.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=170 loss=4689.5181 pred=4686.2363 aux=164.0910 lr=1.70e-05 toks/s=225,372 tflops=399.01 mfu=40.35% step_ms=397.96 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=180 loss=4886.7715 pred=4883.6382 aux=156.6661 lr=1.80e-05 toks/s=225,184 tflops=398.68 mfu=40.31% step_ms=398.42 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=190 loss=4614.6328 pred=4611.4521 aux=159.0338 lr=1.90e-05 toks/s=224,435 tflops=397.35 mfu=40.18% step_ms=399.68 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=200 loss=4769.3843 pred=4766.3506 aux=151.6758 lr=2.00e-05 toks/s=218,539 tflops=386.92 mfu=39.12% step_ms=410.59 sm_util=89.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=210 loss=4679.8340 pred=4676.8154 aux=150.9385 lr=2.10e-05 toks/s=222,954 tflops=394.73 mfu=39.91% step_ms=402.31 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=220 loss=4337.3022 pred=4334.1694 aux=156.6346 lr=2.20e-05 toks/s=224,502 tflops=397.47 mfu=40.19% step_ms=399.60 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=230 loss=4258.0498 pred=4255.0098 aux=151.9912 lr=2.30e-05 toks/s=222,194 tflops=393.39 mfu=39.78% step_ms=403.67 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=240 loss=4304.5454 pred=4301.5190 aux=151.3144 lr=2.40e-05 toks/s=222,673 tflops=394.23 mfu=39.86% step_ms=402.85 sm_util=93.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=250 loss=3899.8667 pred=3896.7329 aux=156.6869 lr=2.50e-05 toks/s=220,660 tflops=390.67 mfu=39.50% step_ms=406.62 sm_util=85.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=260 loss=4344.7314 pred=4341.7793 aux=147.6096 lr=2.60e-05 toks/s=223,450 tflops=395.61 mfu=40.00% step_ms=401.53 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=270 loss=3671.2419 pred=3668.1086 aux=156.6688 lr=2.70e-05 toks/s=222,922 tflops=394.67 mfu=39.91% step_ms=402.50 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=280 loss=3601.3774 pred=3598.2979 aux=153.9836 lr=2.80e-05 toks/s=223,342 tflops=395.42 mfu=39.98% step_ms=401.80 sm_util=93.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=290 loss=3651.3320 pred=3648.1746 aux=157.8784 lr=2.90e-05 toks/s=223,149 tflops=395.08 mfu=39.95% step_ms=402.08 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=300 loss=3381.5122 pred=3378.1963 aux=165.7950 lr=3.00e-05 toks/s=221,201 tflops=391.63 mfu=39.60% step_ms=405.71 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=310 loss=3012.9758 pred=3009.7202 aux=162.7803 lr=3.10e-05 toks/s=224,485 tflops=397.44 mfu=40.19% step_ms=399.85 sm_util=87.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=320 loss=2720.7036 pred=2717.3208 aux=169.1390 lr=3.20e-05 toks/s=227,032 tflops=401.95 mfu=40.64% step_ms=395.27 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=330 loss=3062.9570 pred=3059.5510 aux=170.2958 lr=3.30e-05 toks/s=226,141 tflops=400.37 mfu=40.48% step_ms=396.86 sm_util=81.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=340 loss=3063.4570 pred=3060.2266 aux=161.5252 lr=3.40e-05 toks/s=221,058 tflops=391.37 mfu=39.57% step_ms=405.97 sm_util=98.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=350 loss=3788.5864 pred=3785.2615 aux=166.2507 lr=3.50e-05 toks/s=221,895 tflops=392.86 mfu=39.72% step_ms=404.28 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=360 loss=2653.1885 pred=2649.8250 aux=168.1761 lr=3.60e-05 toks/s=228,619 tflops=404.76 mfu=40.93% step_ms=392.51 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=370 loss=2560.9158 pred=2557.5981 aux=165.8758 lr=3.70e-05 toks/s=222,580 tflops=394.07 mfu=39.85% step_ms=403.21 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=380 loss=1854.7446 pred=1851.2743 aux=173.5165 lr=3.80e-05 toks/s=226,164 tflops=400.42 mfu=40.49% step_ms=396.73 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=390 loss=2068.5515 pred=2065.2815 aux=163.5003 lr=3.90e-05 toks/s=224,591 tflops=397.63 mfu=40.21% step_ms=399.50 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=400 loss=2390.9475 pred=2387.6926 aux=162.7434 lr=4.00e-05 toks/s=223,602 tflops=395.88 mfu=40.03% step_ms=401.27 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=410 loss=1284.8772 pred=1281.5583 aux=165.9419 lr=4.10e-05 toks/s=228,052 tflops=403.76 mfu=40.82% step_ms=393.37 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=420 loss=1922.4474 pred=1919.2623 aux=159.2512 lr=4.20e-05 toks/s=222,751 tflops=394.37 mfu=39.88% step_ms=402.90 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=430 loss=2119.8450 pred=2116.6113 aux=161.6814 lr=4.30e-05 toks/s=222,560 tflops=394.03 mfu=39.84% step_ms=403.03 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=440 loss=2093.0957 pred=2090.0513 aux=152.2259 lr=4.40e-05 toks/s=228,993 tflops=405.42 mfu=40.99% step_ms=391.71 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=450 loss=1670.7167 pred=1667.5272 aux=159.4710 lr=4.50e-05 toks/s=222,853 tflops=394.55 mfu=39.89% step_ms=402.49 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=460 loss=1707.0947 pred=1704.0236 aux=153.5566 lr=4.60e-05 toks/s=225,503 tflops=399.25 mfu=40.37% step_ms=397.84 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=470 loss=1134.1415 pred=1130.9937 aux=157.3930 lr=4.70e-05 toks/s=222,898 tflops=394.63 mfu=39.90% step_ms=402.57 sm_util=93.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=480 loss=528.0311 pred=524.7496 aux=164.0758 lr=4.80e-05 toks/s=181,042 tflops=320.53 mfu=32.41% step_ms=495.92 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=490 loss=765.3400 pred=762.0929 aux=162.3568 lr=4.90e-05 toks/s=223,996 tflops=396.58 mfu=40.10% step_ms=400.59 sm_util=92.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=500 loss=1054.4230 pred=1051.3120 aux=155.5469 lr=5.00e-05 toks/s=221,178 tflops=391.59 mfu=39.59% step_ms=405.70 sm_util=82.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=510 loss=2397.5247 pred=2394.4832 aux=152.0752 lr=5.10e-05 toks/s=219,112 tflops=387.93 mfu=39.22% step_ms=409.56 sm_util=95.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=520 loss=1021.7197 pred=1018.5056 aux=160.7013 lr=5.20e-05 toks/s=224,895 tflops=398.17 mfu=40.26% step_ms=399.03 sm_util=91.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=530 loss=851.9277 pred=848.6572 aux=163.5245 lr=5.30e-05 toks/s=220,409 tflops=390.23 mfu=39.46% step_ms=407.21 sm_util=84.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=540 loss=1978.7312 pred=1975.4739 aux=162.8687 lr=5.40e-05 toks/s=221,264 tflops=391.74 mfu=39.61% step_ms=405.57 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=550 loss=1929.6866 pred=1926.6002 aux=154.3187 lr=5.50e-05 toks/s=216,977 tflops=384.15 mfu=38.84% step_ms=413.55 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=560 loss=726.2624 pred=722.9949 aux=163.3771 lr=5.60e-05 toks/s=229,098 tflops=405.61 mfu=41.01% step_ms=391.51 sm_util=85.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=570 loss=1085.3627 pred=1082.2006 aux=158.1076 lr=5.70e-05 toks/s=225,276 tflops=398.84 mfu=40.33% step_ms=398.29 sm_util=87.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=580 loss=100.0261 pred=96.5609 aux=173.2594 lr=5.80e-05 toks/s=224,913 tflops=398.20 mfu=40.26% step_ms=398.86 sm_util=86.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=590 loss=169.2881 pred=166.1154 aux=158.6370 lr=5.90e-05 toks/s=223,234 tflops=395.23 mfu=39.96% step_ms=402.10 sm_util=84.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=600 loss=297.5385 pred=294.3382 aux=160.0189 lr=6.00e-05 toks/s=229,321 tflops=406.00 mfu=41.05% step_ms=391.21 sm_util=89.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=610 loss=125.5840 pred=122.4160 aux=158.4019 lr=6.10e-05 toks/s=220,288 tflops=390.01 mfu=39.43% step_ms=407.32 sm_util=86.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=620 loss=129.1823 pred=125.9798 aux=160.1221 lr=6.20e-05 toks/s=225,932 tflops=400.00 mfu=40.45% step_ms=397.02 sm_util=84.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=630 loss=254.7453 pred=251.5730 aux=158.6156 lr=6.30e-05 toks/s=226,933 tflops=401.78 mfu=40.62% step_ms=395.17 sm_util=85.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=640 loss=670.1309 pred=667.0147 aux=155.8092 lr=6.40e-05 toks/s=227,962 tflops=403.60 mfu=40.81% step_ms=393.47 sm_util=98.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=650 loss=152.1122 pred=148.8190 aux=164.6629 lr=6.50e-05 toks/s=225,865 tflops=399.89 mfu=40.43% step_ms=397.07 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=660 loss=417.3076 pred=414.1596 aux=157.3998 lr=6.60e-05 toks/s=221,563 tflops=392.27 mfu=39.66% step_ms=404.85 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=670 loss=237.4961 pred=234.2534 aux=162.1338 lr=6.70e-05 toks/s=219,996 tflops=389.49 mfu=39.38% step_ms=407.75 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=680 loss=90.1067 pred=86.7016 aux=170.2560 lr=6.80e-05 toks/s=225,195 tflops=398.70 mfu=40.31% step_ms=398.33 sm_util=97.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=690 loss=182.5011 pred=179.1408 aux=168.0150 lr=6.90e-05 toks/s=227,078 tflops=402.03 mfu=40.65% step_ms=395.17 sm_util=88.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=700 loss=234.0364 pred=230.6399 aux=169.8267 lr=7.00e-05 toks/s=231,566 tflops=409.98 mfu=41.45% step_ms=387.34 sm_util=98.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=710 loss=79.5619 pred=76.2547 aux=165.3598 lr=7.10e-05 toks/s=225,575 tflops=399.37 mfu=40.38% step_ms=397.78 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=720 loss=66.8612 pred=63.5855 aux=163.7833 lr=7.20e-05 toks/s=227,678 tflops=403.10 mfu=40.76% step_ms=393.94 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=730 loss=137.3790 pred=134.1598 aux=160.9600 lr=7.30e-05 toks/s=224,543 tflops=397.54 mfu=40.20% step_ms=399.54 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=740 loss=343.2557 pred=340.1891 aux=153.3291 lr=7.40e-05 toks/s=226,540 tflops=401.08 mfu=40.55% step_ms=396.02 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=750 loss=45.0012 pred=41.7336 aux=163.3813 lr=7.50e-05 toks/s=225,921 tflops=399.98 mfu=40.44% step_ms=397.02 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=760 loss=66.2866 pred=63.1181 aux=158.4252 lr=7.60e-05 toks/s=223,711 tflops=396.07 mfu=40.05% step_ms=400.91 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=770 loss=155.2746 pred=152.1369 aux=156.8853 lr=7.70e-05 toks/s=229,764 tflops=406.79 mfu=41.13% step_ms=390.34 sm_util=90.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=780 loss=91.9211 pred=88.7623 aux=157.9399 lr=7.80e-05 toks/s=222,301 tflops=393.58 mfu=39.80% step_ms=403.74 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=790 loss=240.1355 pred=237.1174 aux=150.9077 lr=7.90e-05 toks/s=224,075 tflops=396.72 mfu=40.11% step_ms=400.58 sm_util=85.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=800 loss=144.9815 pred=141.9607 aux=151.0384 lr=8.00e-05 toks/s=222,245 tflops=393.48 mfu=39.79% step_ms=403.92 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=810 loss=81.1798 pred=78.0268 aux=157.6492 lr=8.10e-05 toks/s=227,005 tflops=401.90 mfu=40.64% step_ms=395.14 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=820 loss=119.4452 pred=116.3787 aux=153.3213 lr=8.20e-05 toks/s=222,243 tflops=393.47 mfu=39.78% step_ms=403.84 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=830 loss=294.1397 pred=291.0806 aux=152.9559 lr=8.30e-05 toks/s=225,118 tflops=398.56 mfu=40.30% step_ms=398.42 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=840 loss=118.8749 pred=115.8671 aux=150.3893 lr=8.40e-05 toks/s=222,172 tflops=393.35 mfu=39.77% step_ms=403.89 sm_util=91.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=850 loss=117.2587 pred=113.9907 aux=163.3988 lr=8.50e-05 toks/s=225,584 tflops=399.39 mfu=40.38% step_ms=397.64 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=860 loss=60.0585 pred=56.7303 aux=166.4096 lr=8.60e-05 toks/s=221,579 tflops=392.30 mfu=39.67% step_ms=405.08 sm_util=84.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=870 loss=367.5660 pred=364.5413 aux=151.2365 lr=8.70e-05 toks/s=219,131 tflops=387.96 mfu=39.23% step_ms=409.50 sm_util=93.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=880 loss=83.1368 pred=79.9024 aux=161.7207 lr=8.80e-05 toks/s=222,613 tflops=394.13 mfu=39.85% step_ms=403.00 sm_util=81.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=890 loss=113.2785 pred=110.1918 aux=154.3357 lr=8.90e-05 toks/s=221,252 tflops=391.72 mfu=39.61% step_ms=405.61 sm_util=85.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=900 loss=133.2361 pred=130.1100 aux=156.3067 lr=9.00e-05 toks/s=224,023 tflops=396.62 mfu=40.10% step_ms=400.52 sm_util=84.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=910 loss=232.7803 pred=229.7670 aux=150.6620 lr=9.10e-05 toks/s=219,641 tflops=388.87 mfu=39.32% step_ms=408.63 sm_util=98.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=920 loss=308.2750 pred=305.0703 aux=160.2303 lr=9.20e-05 toks/s=225,604 tflops=399.42 mfu=40.39% step_ms=397.61 sm_util=98.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=930 loss=121.0736 pred=117.8411 aux=161.6253 lr=9.30e-05 toks/s=224,875 tflops=398.13 mfu=40.26% step_ms=398.89 sm_util=98.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=940 loss=123.7397 pred=120.3638 aux=168.7956 lr=9.40e-05 toks/s=226,621 tflops=401.22 mfu=40.57% step_ms=395.86 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=950 loss=167.4992 pred=164.2918 aux=160.3717 lr=9.50e-05 toks/s=225,799 tflops=399.77 mfu=40.42% step_ms=397.43 sm_util=98.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=960 loss=340.0943 pred=336.9182 aux=158.8030 lr=9.60e-05 toks/s=227,903 tflops=403.49 mfu=40.80% step_ms=393.74 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=970 loss=190.1913 pred=187.0256 aux=158.2856 lr=9.70e-05 toks/s=228,955 tflops=405.36 mfu=40.99% step_ms=391.96 sm_util=91.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=980 loss=274.3357 pred=270.9675 aux=168.4099 lr=9.80e-05 toks/s=222,190 tflops=393.38 mfu=39.78% step_ms=403.79 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=990 loss=379.5789 pred=376.3757 aux=160.1630 lr=9.90e-05 toks/s=225,478 tflops=399.20 mfu=40.36% step_ms=398.03 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=1000 loss=419.0906 pred=415.8816 aux=160.4488 lr=1.00e-04 toks/s=223,751 tflops=396.14 mfu=40.05% step_ms=400.97 sm_util=82.0% hbm_util=54.0% mem_ctrl_util=54.0%
val step=1000 pred=309.1890 aux=139.5640 mae=116.8252 mse=79388.4614
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=74.30
```

So results aren't good, but we can train, it does something! I will revisit to train (TBH, probably there are a lot of model performance tricks I'm sure to improve performance).
