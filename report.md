# Experiment Report

Generated from logs in `experiments/`.


## Summary

| Experiment | Toks/Sec | TFLOPS | MFU | Step ms | Aux Loss | SM Util | HBM Util | Mem Ctrl Util | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|
| dense_compute_matched_moe_active_ | 563,605 | 151.73 | 15.34 | 232.35 | 144.0000 | 80.0 | 54.0 | 54.0 | 40.25 |
| dense_ffn_bf16_flash_ | 475,124 | 127.91 | 12.93 | 275.50 | 144.0000 | 87.0 | 54.0 | 54.0 | 39.68 |
| dense_ffn_bf16_flash_compile_ | 574,063 | 154.54 | 15.63 | 228.15 | 144.0000 | 72.0 | 47.0 | 47.0 | 32.57 |
| dense_ffn_bf16_sdpa_ | 245,148 | 66.00 | 6.67 | 532.60 |  | 99.0 | 29.0 | 29.0 | 45.93 |
| logical_dense_cap_0_9_ | 380,824 | 194.80 | 19.70 | 343.45 | 143.5254 | 92.0 | 60.0 | 60.0 | 53.54 |
| logical_dense_cap_1_1_ | 365,751 | 196.41 | 19.86 | 357.63 | 143.6436 | 88.0 | 55.0 | 55.0 | 58.03 |
| logical_dense_cap_1_3_ | 353,074 | 198.60 | 20.08 | 370.50 | 143.8579 | 98.0 | 70.0 | 70.0 | 62.59 |
| logical_dense_cap_1_5_ | 339,824 | 199.80 | 20.20 | 384.98 | 143.8817 | 96.0 | 69.0 | 69.0 | 66.55 |
| moe_naive_bf16_ | 251,783 | 67.78 | 6.85 | 520.32 | 144.0010 | 52.0 | 35.0 | 35.0 | 59.90 |
| moe_naive_bf16_compile_ | 245,329 | 66.04 | 6.68 | 534.14 | 144.1115 | 41.0 | 29.0 | 29.0 | 50.81 |
| moe_onehot_bf16_ | 241,247 | 64.95 | 6.57 | 543.14 | 144.0026 | 58.0 | 38.0 | 38.0 | 55.47 |
| moe_onehot_bf16_compile_ | 234,108 | 63.02 | 6.37 | 559.76 | 144.1072 | 37.0 | 25.0 | 25.0 | 47.16 |
| moe_scatter_bf16_ | 353,145 | 198.64 | 20.08 | 370.41 | 143.8569 | 96.0 | 71.0 | 71.0 | 62.59 |
| moe_scatter_bf16_compile_ | 431,195 | 242.54 | 24.52 | 303.47 | 145.3451 | 97.0 | 68.0 | 68.0 | 51.68 |
| per_expert_tile_128_ | 469,466 | 126.38 | 12.78 | 279.06 | 144.1190 | 74.0 | 49.0 | 49.0 | 50.78 |
| per_expert_tile_1_ | 243,009 | 65.42 | 6.61 | 539.25 | 144.1054 | 41.0 | 28.0 | 28.0 | 50.81 |
| per_expert_tile_64_ | 453,473 | 122.08 | 12.34 | 288.92 | 144.1259 | 69.0 | 46.0 | 46.0 | 50.78 |
| scatter_compile_cap_0_9_ | 462,464 | 236.56 | 23.92 | 282.92 | 145.5171 | 90.0 | 61.0 | 61.0 | 44.58 |
| scatter_compile_cap_1_1_ | 446,025 | 239.52 | 24.22 | 293.37 | 145.7734 | 96.0 | 67.0 | 67.0 | 48.08 |
| scatter_compile_cap_1_5_ | 414,306 | 243.60 | 24.63 | 315.88 | 144.8360 | 79.0 | 54.0 | 54.0 | 55.09 |
| scatter_compile_tile_128_ | 430,133 | 241.94 | 24.46 | 304.24 | 145.3324 | 96.0 | 68.0 | 68.0 | 51.79 |
| scatter_compile_tile_64_ | 431,353 | 242.63 | 24.53 | 303.37 | 145.3510 | 89.0 | 62.0 | 62.0 | 51.79 |


## Best by Family

| Experiment Family | Best Variant | Toks/Sec | TFLOPS | MFU | Step ms | Peak VRAM (GB) | Aux loss | SM util | HBM util |
|---|---|---|---|---|---|---|---|---|---|
| Dense FFN | dense_compute_matched_moe_active | 563,605 | 151.73 | 15.34 | 232.35 | 40.25 | 144.0000 | 80.0 | 54.0 |
| Per-expert  | moe_naive_bf16_ | 251,783 | 67.78 | 6.85 | 520.32 | 59.90 | 144.0010 | 52.0 | 35.0 |
| MoE Onehot | moe_onehot_bf16_ | 241,247 | 64.95 | 6.57 | 543.14 | 55.47 | 144.0026 | 58.0 | 38.0 |
| Logical Batched MoE | moe_scatter_bf16_compile_ | 431,195 | 242.54 | 24.52 | 303.47 | 51.68 | 145.3451 | 97.0 | 68.0 |
| Per-expert w/ token rounding| per_expert_tile_128_ | 469,466 | 126.38 | 12.78 | 279.06 | 50.78 | 144.1190 | 74.0 | 49.0 |
| Logical Dense (cap) | scatter_compile_cap_0_9_ | 462,464 | 236.56 | 23.92 | 282.92 | 44.58 | 145.5171 | 90.0 | 61.0 |
| Logical Dense (rounding + cap) | scatter_compile_tile_64_ | 431,353 | 242.63 | 24.53 | 303.37 | 51.79 | 145.3510 | 89.0 | 62.0 |

All methods are compiled

| Exp name                                       |                  toks/sec | max VRAM (GB) | Time to 4,194,304 tokens | Est. time on 128 GPUs (linear) |
| ---------------------------------------------- | ------------------------: | ------------: | -----------------------: | -----------------------------: |
| **Per-expert w/ rounding** (moe-impl=standard) |               **522,598** |     **76.61** |               **8.03 s** |                   **0.0627 s** |
| **OneHot** (moe-impl=onehot)                   |               **321,153** |     **76.21** |              **13.06 s** |                   **0.1020 s** |
| **Paper reference** (128 GPUs)                 | **≈ 4,993,219** (derived) |           N/A |     **0.84 s** *(given)* |                     **0.84 s** |



## Max Batch Size , per expert rounding

```
((py312) ) ubuntu@192-222-54-234:~/FoundationTS$ foundationts train --dataset-path time300b_selected --steps-per-epoch 80 --epochs 1 --batch-size 56 --seq-max-len 4096 --seq-stride 4096 --num-expert
-layers 1 --hidden-size 384 --n-head 12 --n-decoder-layers 12 --num-experts 8 --k 2 --d-ff 1536 --d-expert 768 --attn-backend flash --log-every 10 --checkpoint-every 0 --log-perf-metrics --mfu-peak-tflops 989 --moe-impl standard --moe-m-tile 128 --compile 
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.9304 pred=0.5815 aux=167.4463 lr=1.00e-06 toks/s=100,504 tflops=27.06 mfu=2.74% step_ms=2112.23 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0%
step=20 loss=3.7498 pred=0.5620 aux=159.3932 lr=2.00e-06 toks/s=520,874 tflops=140.22 mfu=14.18% step_ms=439.85 sm_util=95.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=30 loss=3.5658 pred=0.4928 aux=153.6483 lr=3.00e-06 toks/s=502,600 tflops=135.30 mfu=13.68% step_ms=455.89 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=40 loss=3.3385 pred=0.4237 aux=145.7430 lr=4.00e-06 toks/s=518,809 tflops=139.67 mfu=14.12% step_ms=441.55 sm_util=99.0% hbm_util=73.0% mem_ctrl_util=73.0%
step=50 loss=3.3310 pred=0.4171 aux=145.6949 lr=5.00e-06 toks/s=529,108 tflops=142.44 mfu=14.40% step_ms=432.99 sm_util=91.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=60 loss=3.2564 pred=0.3691 aux=144.3674 lr=6.00e-06 toks/s=531,631 tflops=143.12 mfu=14.47% step_ms=430.86 sm_util=99.0% hbm_util=75.0% mem_ctrl_util=75.0%
step=70 loss=3.2612 pred=0.3548 aux=145.3193 lr=7.00e-06 toks/s=525,210 tflops=141.39 mfu=14.30% step_ms=436.31 sm_util=99.0% hbm_util=74.0% mem_ctrl_util=74.0%
step=80 loss=3.1749 pred=0.2934 aux=144.0757 lr=8.00e-06 toks/s=529,953 tflops=142.67 mfu=14.43% step_ms=432.34 sm_util=95.0% hbm_util=68.0% mem_ctrl_util=68.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=76.61
```

## Max Batch size, one hot
```
((py312) ) ubuntu@192-222-54-234:~/FoundationTS$ foundationts train --dataset-path time300b_selected --steps-per-epoch 80 --epochs 1 --batch-size 48 --seq-max-len 4096 --seq-stride 4096 --num-expert-layers 1 --hidden-size 384 --n-head 12 --n-decoder-layers 12 --
num-experts 8 --k 2 --d-ff 1536 --d-expert 768 --attn-backend flash --log-every 10 --checkpoint-every 0 --log-perf-metrics --mfu-peak-tflops 989 --moe-impl onehot
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6178 pred=0.5762 aux=152.0805 lr=1.00e-06 toks/s=203,736 tflops=54.85 mfu=5.55% step_ms=796.56 sm_util=71.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=20 loss=3.5320 pred=0.5731 aux=147.9424 lr=2.00e-06 toks/s=316,168 tflops=85.11 mfu=8.61% step_ms=620.75 sm_util=69.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=30 loss=3.3637 pred=0.4837 aux=144.0024 lr=3.00e-06 toks/s=318,897 tflops=85.85 mfu=8.68% step_ms=615.86 sm_util=75.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=40 loss=3.3016 pred=0.4404 aux=143.0591 lr=4.00e-06 toks/s=316,817 tflops=85.29 mfu=8.62% step_ms=619.93 sm_util=78.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=50 loss=3.2930 pred=0.3797 aux=145.6655 lr=5.00e-06 toks/s=322,364 tflops=86.78 mfu=8.77% step_ms=608.79 sm_util=76.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=60 loss=3.2224 pred=0.3383 aux=144.2039 lr=6.00e-06 toks/s=324,288 tflops=87.30 mfu=8.83% step_ms=605.62 sm_util=73.0% hbm_util=48.0% mem_ctrl_util=48.0%
step=70 loss=3.2126 pred=0.3444 aux=143.4094 lr=7.00e-06 toks/s=326,288 tflops=87.84 mfu=8.88% step_ms=601.78 sm_util=60.0% hbm_util=41.0% mem_ctrl_util=41.0%
step=80 loss=3.1795 pred=0.3000 aux=143.9715 lr=8.00e-06 toks/s=323,246 tflops=87.02 mfu=8.80% step_ms=607.44 sm_util=64.0% hbm_util=45.0% mem_ctrl_util=45.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=76.21

```




## Notes
Default parameters:

```
foundationts train
--dataset-path time300b_selected
--steps-per-epoch 80
--epochs 1
--batch-size 32
--seq-max-len 4096
--seq-stride 4096
--num-expert-layers 1
--hidden-size 384
--n-head 12
--n-decoder-layers 12
--num-experts 8
--k 2
--d-ff 1536
--d-expert 768
--attn-backend flash
--log-every 10
--checkpoint-every 0
--log-perf-metrics
--mfu-peak-tflops 989
```

Standard settings: params_total=113.35M, params_active=49.65M, precision=bf16.
Models breaking the standard settings:
- dense_compute_matched_moe_active_: params_total=49.62M, params_active=49.62M, precision=bf16
- dense_ffn_bf16_flash_: params_total=39.00M, params_active=39.00M, precision=bf16
- dense_ffn_bf16_flash_compile_: params_total=39.00M, params_active=39.00M, precision=bf16
- dense_ffn_bf16_sdpa_: params_total=39.00M, params_active=39.00M, precision=bf16


## dense_compute_matched_moe_active_

```
params total=49.62M (49,618,665) active=49.62M (49,618,665)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4973 pred=0.6173 aux=144.0000 lr=1.00e-06 toks/s=124,617 tflops=33.55 mfu=3.39% step_ms=876.33 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=20 loss=3.4575 pred=0.5775 aux=144.0000 lr=2.00e-06 toks/s=559,988 tflops=150.75 mfu=15.24% step_ms=233.69 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=30 loss=3.3322 pred=0.4522 aux=144.0000 lr=3.00e-06 toks/s=559,036 tflops=150.50 mfu=15.22% step_ms=234.24 sm_util=82.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=40 loss=3.2906 pred=0.4106 aux=144.0000 lr=4.00e-06 toks/s=548,645 tflops=147.70 mfu=14.93% step_ms=238.79 sm_util=70.0% hbm_util=47.0% mem_ctrl_util=47.0%
step=50 loss=3.2531 pred=0.3731 aux=144.0000 lr=5.00e-06 toks/s=562,904 tflops=151.54 mfu=15.32% step_ms=232.55 sm_util=84.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.1874 pred=0.3074 aux=144.0000 lr=6.00e-06 toks/s=559,549 tflops=150.63 mfu=15.23% step_ms=234.02 sm_util=94.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=70 loss=3.2074 pred=0.3274 aux=144.0000 lr=7.00e-06 toks/s=544,652 tflops=146.62 mfu=14.83% step_ms=240.41 sm_util=94.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=80 loss=3.1668 pred=0.2868 aux=144.0000 lr=8.00e-06 toks/s=563,605 tflops=151.73 mfu=15.34% step_ms=232.35 sm_util=80.0% hbm_util=54.0% mem_ctrl_util=54.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=40.25
```
## dense_ffn_bf16_flash_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5331 pred=0.6531 aux=144.0000 lr=1.00e-06 toks/s=228,369 tflops=61.48 mfu=6.22% step_ms=427.66 sm_util=99.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=3.4422 pred=0.5622 aux=144.0000 lr=2.00e-06 toks/s=472,337 tflops=127.16 mfu=12.86% step_ms=277.07 sm_util=86.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=30 loss=3.3260 pred=0.4460 aux=144.0000 lr=3.00e-06 toks/s=471,729 tflops=126.99 mfu=12.84% step_ms=277.42 sm_util=91.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=3.3151 pred=0.4351 aux=144.0000 lr=4.00e-06 toks/s=478,768 tflops=128.89 mfu=13.03% step_ms=273.38 sm_util=94.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=50 loss=3.2279 pred=0.3479 aux=144.0000 lr=5.00e-06 toks/s=481,526 tflops=129.63 mfu=13.11% step_ms=271.68 sm_util=88.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=60 loss=3.2051 pred=0.3251 aux=144.0000 lr=6.00e-06 toks/s=477,066 tflops=128.43 mfu=12.99% step_ms=274.36 sm_util=92.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=3.1957 pred=0.3157 aux=144.0000 lr=7.00e-06 toks/s=491,420 tflops=132.29 mfu=13.38% step_ms=266.24 sm_util=90.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=80 loss=3.2114 pred=0.3314 aux=144.0000 lr=8.00e-06 toks/s=475,124 tflops=127.91 mfu=12.93% step_ms=275.50 sm_util=87.0% hbm_util=54.0% mem_ctrl_util=54.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=39.68
```
## dense_ffn_bf16_flash_compile_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5331 pred=0.6531 aux=144.0000 lr=1.00e-06 toks/s=105,597 tflops=28.43 mfu=2.87% step_ms=1073.90 sm_util=77.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=20 loss=3.4420 pred=0.5620 aux=144.0000 lr=2.00e-06 toks/s=569,882 tflops=153.42 mfu=15.51% step_ms=229.65 sm_util=84.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=30 loss=3.3258 pred=0.4458 aux=144.0000 lr=3.00e-06 toks/s=568,483 tflops=153.04 mfu=15.47% step_ms=230.36 sm_util=77.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=40 loss=3.3149 pred=0.4349 aux=144.0000 lr=4.00e-06 toks/s=581,822 tflops=156.63 mfu=15.84% step_ms=225.12 sm_util=72.0% hbm_util=47.0% mem_ctrl_util=47.0%
step=50 loss=3.2278 pred=0.3478 aux=144.0000 lr=5.00e-06 toks/s=586,028 tflops=157.76 mfu=15.95% step_ms=223.35 sm_util=87.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=3.2049 pred=0.3249 aux=144.0000 lr=6.00e-06 toks/s=577,392 tflops=155.44 mfu=15.72% step_ms=226.81 sm_util=91.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=70 loss=3.1957 pred=0.3157 aux=144.0000 lr=7.00e-06 toks/s=596,702 tflops=160.64 mfu=16.24% step_ms=219.56 sm_util=88.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=3.2114 pred=0.3314 aux=144.0000 lr=8.00e-06 toks/s=574,063 tflops=154.54 mfu=15.63% step_ms=228.15 sm_util=72.0% hbm_util=47.0% mem_ctrl_util=47.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=32.57
```
## dense_ffn_bf16_sdpa_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=nan pred=nan aux=nan lr=1.00e-06 toks/s=155,788 tflops=41.94 mfu=4.24% step_ms=695.39 sm_util=99.0% hbm_util=29.0% mem_ctrl_util=29.0%
step=20 loss=nan pred=nan aux=nan lr=2.00e-06 toks/s=244,296 tflops=65.77 mfu=6.65% step_ms=534.48 sm_util=99.0% hbm_util=33.0% mem_ctrl_util=33.0%
step=30 loss=nan pred=nan aux=nan lr=3.00e-06 toks/s=244,380 tflops=65.79 mfu=6.65% step_ms=534.29 sm_util=93.0% hbm_util=30.0% mem_ctrl_util=30.0%
step=40 loss=nan pred=nan aux=nan lr=4.00e-06 toks/s=245,014 tflops=65.96 mfu=6.67% step_ms=532.91 sm_util=99.0% hbm_util=33.0% mem_ctrl_util=33.0%
step=50 loss=nan pred=nan aux=nan lr=5.00e-06 toks/s=245,836 tflops=66.18 mfu=6.69% step_ms=531.13 sm_util=100.0% hbm_util=26.0% mem_ctrl_util=26.0%
step=60 loss=nan pred=nan aux=nan lr=6.00e-06 toks/s=245,313 tflops=66.04 mfu=6.68% step_ms=532.27 sm_util=92.0% hbm_util=27.0% mem_ctrl_util=27.0%
step=70 loss=nan pred=nan aux=nan lr=7.00e-06 toks/s=246,649 tflops=66.40 mfu=6.71% step_ms=529.35 sm_util=100.0% hbm_util=26.0% mem_ctrl_util=26.0%
step=80 loss=nan pred=nan aux=nan lr=8.00e-06 toks/s=245,148 tflops=66.00 mfu=6.67% step_ms=532.60 sm_util=99.0% hbm_util=29.0% mem_ctrl_util=29.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=45.93
```
## logical_dense_cap_0_9_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4996 pred=0.5874 aux=145.6129 lr=1.00e-06 toks/s=197,750 tflops=101.15 mfu=10.23% step_ms=512.91 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=20 loss=3.3997 pred=0.5050 aux=144.7324 lr=2.00e-06 toks/s=382,114 tflops=195.46 mfu=19.76% step_ms=342.25 sm_util=89.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=30 loss=3.3930 pred=0.4994 aux=144.6786 lr=3.00e-06 toks/s=380,223 tflops=194.49 mfu=19.67% step_ms=343.89 sm_util=87.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=40 loss=3.3421 pred=0.4657 aux=143.8194 lr=4.00e-06 toks/s=382,508 tflops=195.66 mfu=19.78% step_ms=341.99 sm_util=87.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=50 loss=3.2622 pred=0.3847 aux=143.8763 lr=5.00e-06 toks/s=388,857 tflops=198.91 mfu=20.11% step_ms=336.32 sm_util=86.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=60 loss=3.2466 pred=0.3626 aux=144.1985 lr=6.00e-06 toks/s=386,400 tflops=197.65 mfu=19.99% step_ms=338.57 sm_util=84.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=70 loss=3.2401 pred=0.3528 aux=144.3647 lr=7.00e-06 toks/s=379,771 tflops=194.26 mfu=19.64% step_ms=344.37 sm_util=88.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=80 loss=3.2121 pred=0.3416 aux=143.5254 lr=8.00e-06 toks/s=380,824 tflops=194.80 mfu=19.70% step_ms=343.45 sm_util=92.0% hbm_util=60.0% mem_ctrl_util=60.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=53.54
```
## logical_dense_cap_1_1_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4986 pred=0.5873 aux=145.5670 lr=1.00e-06 toks/s=192,868 tflops=103.57 mfu=10.47% step_ms=527.53 sm_util=98.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=20 loss=3.3963 pred=0.5048 aux=144.5746 lr=2.00e-06 toks/s=367,227 tflops=197.20 mfu=19.94% step_ms=356.17 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=30 loss=3.3897 pred=0.4990 aux=144.5356 lr=3.00e-06 toks/s=365,056 tflops=196.04 mfu=19.82% step_ms=358.22 sm_util=98.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=40 loss=3.3434 pred=0.4652 aux=143.9120 lr=4.00e-06 toks/s=367,040 tflops=197.10 mfu=19.93% step_ms=356.43 sm_util=87.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=50 loss=3.2597 pred=0.3843 aux=143.7685 lr=5.00e-06 toks/s=372,863 tflops=200.23 mfu=20.25% step_ms=350.78 sm_util=88.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=60 loss=3.2425 pred=0.3622 aux=144.0142 lr=6.00e-06 toks/s=370,683 tflops=199.06 mfu=20.13% step_ms=353.05 sm_util=98.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=70 loss=3.2387 pred=0.3524 aux=144.3136 lr=7.00e-06 toks/s=364,856 tflops=195.93 mfu=19.81% step_ms=358.46 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=80 loss=3.2137 pred=0.3408 aux=143.6436 lr=8.00e-06 toks/s=365,751 tflops=196.41 mfu=19.86% step_ms=357.63 sm_util=88.0% hbm_util=55.0% mem_ctrl_util=55.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.03
```
## logical_dense_cap_1_3_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4974 pred=0.5873 aux=145.5064 lr=1.00e-06 toks/s=188,904 tflops=106.26 mfu=10.74% step_ms=541.69 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=20 loss=3.3946 pred=0.5047 aux=144.4920 lr=2.00e-06 toks/s=354,526 tflops=199.41 mfu=20.16% step_ms=368.94 sm_util=91.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=30 loss=3.3857 pred=0.4988 aux=144.3455 lr=3.00e-06 toks/s=352,731 tflops=198.40 mfu=20.06% step_ms=370.79 sm_util=98.0% hbm_util=72.0% mem_ctrl_util=72.0%
step=40 loss=3.3465 pred=0.4650 aux=144.0738 lr=4.00e-06 toks/s=354,849 tflops=199.60 mfu=20.18% step_ms=368.69 sm_util=89.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=50 loss=3.2617 pred=0.3842 aux=143.8765 lr=5.00e-06 toks/s=360,222 tflops=202.62 mfu=20.49% step_ms=363.12 sm_util=86.0% hbm_util=55.0% mem_ctrl_util=55.0%
step=60 loss=3.2397 pred=0.3621 aux=143.8802 lr=6.00e-06 toks/s=357,592 tflops=201.14 mfu=20.34% step_ms=365.89 sm_util=98.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=70 loss=3.2379 pred=0.3521 aux=144.2901 lr=7.00e-06 toks/s=352,045 tflops=198.02 mfu=20.02% step_ms=371.52 sm_util=85.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=80 loss=3.2177 pred=0.3406 aux=143.8579 lr=8.00e-06 toks/s=353,074 tflops=198.60 mfu=20.08% step_ms=370.50 sm_util=98.0% hbm_util=70.0% mem_ctrl_util=70.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=62.59
```
## logical_dense_cap_1_5_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4978 pred=0.5873 aux=145.5253 lr=1.00e-06 toks/s=185,288 tflops=108.94 mfu=11.02% step_ms=554.92 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0%
step=20 loss=3.3936 pred=0.5049 aux=144.4386 lr=2.00e-06 toks/s=341,313 tflops=200.68 mfu=20.29% step_ms=383.26 sm_util=99.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=30 loss=3.3832 pred=0.4989 aux=144.2181 lr=3.00e-06 toks/s=339,479 tflops=199.60 mfu=20.18% step_ms=385.27 sm_util=89.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=3.3481 pred=0.4652 aux=144.1430 lr=4.00e-06 toks/s=341,320 tflops=200.68 mfu=20.29% step_ms=383.33 sm_util=85.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=50 loss=3.2637 pred=0.3845 aux=143.9583 lr=5.00e-06 toks/s=346,514 tflops=203.74 mfu=20.60% step_ms=377.51 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=3.2388 pred=0.3622 aux=143.8262 lr=6.00e-06 toks/s=344,050 tflops=202.29 mfu=20.45% step_ms=380.32 sm_util=86.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=3.2394 pred=0.3521 aux=144.3649 lr=7.00e-06 toks/s=338,761 tflops=199.18 mfu=20.14% step_ms=386.13 sm_util=99.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=80 loss=3.2182 pred=0.3406 aux=143.8817 lr=8.00e-06 toks/s=339,824 tflops=199.80 mfu=20.20% step_ms=384.98 sm_util=96.0% hbm_util=69.0% mem_ctrl_util=69.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=66.55
```
## moe_naive_bf16_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6192 pred=0.5798 aux=151.9693 lr=1.00e-06 toks/s=152,165 tflops=40.96 mfu=4.14% step_ms=691.39 sm_util=41.0% hbm_util=29.0% mem_ctrl_util=29.0%
step=20 loss=3.5134 pred=0.5526 aux=148.0428 lr=2.00e-06 toks/s=240,213 tflops=64.67 mfu=6.54% step_ms=545.26 sm_util=64.0% hbm_util=43.0% mem_ctrl_util=43.0%
step=30 loss=3.3970 pred=0.5191 aux=143.8947 lr=3.00e-06 toks/s=246,794 tflops=66.44 mfu=6.72% step_ms=530.73 sm_util=53.0% hbm_util=36.0% mem_ctrl_util=36.0%
step=40 loss=3.3153 pred=0.4569 aux=142.9195 lr=4.00e-06 toks/s=239,522 tflops=64.48 mfu=6.52% step_ms=546.92 sm_util=50.0% hbm_util=35.0% mem_ctrl_util=35.0%
step=50 loss=3.3169 pred=0.4029 aux=145.7003 lr=5.00e-06 toks/s=250,690 tflops=67.49 mfu=6.82% step_ms=522.59 sm_util=60.0% hbm_util=42.0% mem_ctrl_util=42.0%
step=60 loss=3.2235 pred=0.3390 aux=144.2264 lr=6.00e-06 toks/s=249,026 tflops=67.04 mfu=6.78% step_ms=526.09 sm_util=58.0% hbm_util=41.0% mem_ctrl_util=41.0%
step=70 loss=3.1916 pred=0.3214 aux=143.5107 lr=7.00e-06 toks/s=249,630 tflops=67.20 mfu=6.79% step_ms=524.77 sm_util=55.0% hbm_util=37.0% mem_ctrl_util=37.0%
step=80 loss=3.2031 pred=0.3231 aux=144.0010 lr=8.00e-06 toks/s=251,783 tflops=67.78 mfu=6.85% step_ms=520.32 sm_util=52.0% hbm_util=35.0% mem_ctrl_util=35.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.90
```
## moe_naive_bf16_compile_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8932 pred=0.5797 aux=165.6723 lr=1.00e-06 toks/s=89,322 tflops=24.05 mfu=2.43% step_ms=1298.13 sm_util=28.0% hbm_util=19.0% mem_ctrl_util=19.0%
step=20 loss=3.7228 pred=0.5525 aux=158.5180 lr=2.00e-06 toks/s=234,039 tflops=63.00 mfu=6.37% step_ms=559.69 sm_util=37.0% hbm_util=24.0% mem_ctrl_util=24.0%
step=30 loss=3.5805 pred=0.5189 aux=153.0769 lr=3.00e-06 toks/s=238,293 tflops=64.15 mfu=6.49% step_ms=549.88 sm_util=55.0% hbm_util=37.0% mem_ctrl_util=37.0%
step=40 loss=3.3674 pred=0.4567 aux=145.5394 lr=4.00e-06 toks/s=232,155 tflops=62.50 mfu=6.32% step_ms=564.46 sm_util=37.0% hbm_util=25.0% mem_ctrl_util=25.0%
step=50 loss=3.3171 pred=0.4026 aux=145.7240 lr=5.00e-06 toks/s=243,824 tflops=65.64 mfu=6.64% step_ms=537.44 sm_util=47.0% hbm_util=31.0% mem_ctrl_util=31.0%
step=60 loss=3.2249 pred=0.3389 aux=144.2997 lr=6.00e-06 toks/s=240,694 tflops=64.80 mfu=6.55% step_ms=544.43 sm_util=43.0% hbm_util=29.0% mem_ctrl_util=29.0%
step=70 loss=3.2073 pred=0.3213 aux=144.2974 lr=7.00e-06 toks/s=242,670 tflops=65.33 mfu=6.61% step_ms=539.95 sm_util=39.0% hbm_util=27.0% mem_ctrl_util=27.0%
step=80 loss=3.2052 pred=0.3230 aux=144.1115 lr=8.00e-06 toks/s=245,329 tflops=66.04 mfu=6.68% step_ms=534.14 sm_util=41.0% hbm_util=29.0% mem_ctrl_util=29.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=50.81
```
## moe_onehot_bf16_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6192 pred=0.5798 aux=151.9698 lr=1.00e-06 toks/s=148,964 tflops=40.10 mfu=4.05% step_ms=710.56 sm_util=43.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=20 loss=3.5134 pred=0.5526 aux=148.0438 lr=2.00e-06 toks/s=231,527 tflops=62.33 mfu=6.30% step_ms=565.74 sm_util=60.0% hbm_util=40.0% mem_ctrl_util=40.0%
step=30 loss=3.3970 pred=0.5191 aux=143.8946 lr=3.00e-06 toks/s=232,066 tflops=62.47 mfu=6.32% step_ms=564.47 sm_util=58.0% hbm_util=36.0% mem_ctrl_util=36.0%
step=40 loss=3.3153 pred=0.4569 aux=142.9201 lr=4.00e-06 toks/s=232,287 tflops=62.53 mfu=6.32% step_ms=564.00 sm_util=54.0% hbm_util=36.0% mem_ctrl_util=36.0%
step=50 loss=3.3169 pred=0.4029 aux=145.7020 lr=5.00e-06 toks/s=237,561 tflops=63.95 mfu=6.47% step_ms=551.47 sm_util=58.0% hbm_util=36.0% mem_ctrl_util=36.0%
step=60 loss=3.2236 pred=0.3390 aux=144.2289 lr=6.00e-06 toks/s=243,350 tflops=65.51 mfu=6.62% step_ms=538.49 sm_util=59.0% hbm_util=39.0% mem_ctrl_util=39.0%
step=70 loss=3.1918 pred=0.3214 aux=143.5204 lr=7.00e-06 toks/s=237,208 tflops=63.86 mfu=6.46% step_ms=552.23 sm_util=43.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=80 loss=3.2031 pred=0.3231 aux=144.0026 lr=8.00e-06 toks/s=241,247 tflops=64.95 mfu=6.57% step_ms=543.14 sm_util=58.0% hbm_util=38.0% mem_ctrl_util=38.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=55.47
```
## moe_onehot_bf16_compile_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8932 pred=0.5797 aux=165.6730 lr=1.00e-06 toks/s=100,525 tflops=27.06 mfu=2.74% step_ms=1134.40 sm_util=29.0% hbm_util=20.0% mem_ctrl_util=20.0%
step=20 loss=3.7229 pred=0.5525 aux=158.5214 lr=2.00e-06 toks/s=225,881 tflops=60.81 mfu=6.15% step_ms=579.94 sm_util=43.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=30 loss=3.5805 pred=0.5189 aux=153.0767 lr=3.00e-06 toks/s=227,641 tflops=61.28 mfu=6.20% step_ms=575.63 sm_util=41.0% hbm_util=26.0% mem_ctrl_util=26.0%
step=40 loss=3.3673 pred=0.4566 aux=145.5336 lr=4.00e-06 toks/s=191,841 tflops=51.64 mfu=5.22% step_ms=683.10 sm_util=38.0% hbm_util=25.0% mem_ctrl_util=25.0%
step=50 loss=3.3168 pred=0.4026 aux=145.7099 lr=5.00e-06 toks/s=232,925 tflops=62.70 mfu=6.34% step_ms=562.59 sm_util=49.0% hbm_util=32.0% mem_ctrl_util=32.0%
step=60 loss=3.2249 pred=0.3389 aux=144.3004 lr=6.00e-06 toks/s=232,375 tflops=62.56 mfu=6.33% step_ms=563.92 sm_util=46.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=70 loss=3.2071 pred=0.3213 aux=144.2918 lr=7.00e-06 toks/s=232,981 tflops=62.72 mfu=6.34% step_ms=562.44 sm_util=37.0% hbm_util=25.0% mem_ctrl_util=25.0%
step=80 loss=3.2051 pred=0.3230 aux=144.1072 lr=8.00e-06 toks/s=234,108 tflops=63.02 mfu=6.37% step_ms=559.76 sm_util=37.0% hbm_util=25.0% mem_ctrl_util=25.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=47.16
```
## moe_scatter_bf16_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.4974 pred=0.5873 aux=145.5075 lr=1.00e-06 toks/s=188,847 tflops=106.22 mfu=10.74% step_ms=540.78 sm_util=98.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=20 loss=3.3946 pred=0.5047 aux=144.4916 lr=2.00e-06 toks/s=354,867 tflops=199.61 mfu=20.18% step_ms=368.59 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=30 loss=3.3856 pred=0.4988 aux=144.3425 lr=3.00e-06 toks/s=353,209 tflops=198.67 mfu=20.09% step_ms=370.26 sm_util=98.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=40 loss=3.3466 pred=0.4650 aux=144.0781 lr=4.00e-06 toks/s=355,254 tflops=199.82 mfu=20.20% step_ms=368.27 sm_util=90.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=50 loss=3.2617 pred=0.3842 aux=143.8774 lr=5.00e-06 toks/s=360,889 tflops=202.99 mfu=20.53% step_ms=362.45 sm_util=85.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.2397 pred=0.3621 aux=143.8809 lr=6.00e-06 toks/s=358,925 tflops=201.89 mfu=20.41% step_ms=364.52 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=70 loss=3.2376 pred=0.3521 aux=144.2751 lr=7.00e-06 toks/s=352,892 tflops=198.50 mfu=20.07% step_ms=370.63 sm_util=91.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=80 loss=3.2177 pred=0.3406 aux=143.8569 lr=8.00e-06 toks/s=353,145 tflops=198.64 mfu=20.08% step_ms=370.41 sm_util=96.0% hbm_util=71.0% mem_ctrl_util=71.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=62.59
```
## moe_scatter_bf16_compile_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5532 pred=0.5872 aux=148.2987 lr=1.00e-06 toks/s=138,741 tflops=78.04 mfu=7.89% step_ms=773.92 sm_util=95.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=3.4388 pred=0.5046 aux=146.7112 lr=2.00e-06 toks/s=432,765 tflops=243.42 mfu=24.61% step_ms=302.34 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=30 loss=3.3991 pred=0.4985 aux=145.0273 lr=3.00e-06 toks/s=429,306 tflops=241.48 mfu=24.42% step_ms=304.75 sm_util=78.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=40 loss=3.3504 pred=0.4646 aux=144.2863 lr=4.00e-06 toks/s=433,002 tflops=243.56 mfu=24.63% step_ms=302.21 sm_util=74.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=50 loss=3.2653 pred=0.3838 aux=144.0744 lr=5.00e-06 toks/s=440,837 tflops=247.96 mfu=25.07% step_ms=296.81 sm_util=76.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=60 loss=3.2509 pred=0.3618 aux=144.4572 lr=6.00e-06 toks/s=437,750 tflops=246.23 mfu=24.90% step_ms=299.02 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=70 loss=3.2376 pred=0.3519 aux=144.2824 lr=7.00e-06 toks/s=429,577 tflops=241.63 mfu=24.43% step_ms=304.56 sm_util=77.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=80 loss=3.2472 pred=0.3403 aux=145.3451 lr=8.00e-06 toks/s=431,195 tflops=242.54 mfu=24.52% step_ms=303.47 sm_util=97.0% hbm_util=68.0% mem_ctrl_util=68.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=51.68
```
## per_expert_tile_128_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8949 pred=0.5797 aux=165.7569 lr=1.00e-06 toks/s=99,269 tflops=26.72 mfu=2.70% step_ms=1150.94 sm_util=64.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=20 loss=3.7229 pred=0.5525 aux=158.5200 lr=2.00e-06 toks/s=397,320 tflops=106.96 mfu=10.82% step_ms=329.55 sm_util=66.0% hbm_util=43.0% mem_ctrl_util=43.0%
step=30 loss=3.5810 pred=0.5189 aux=153.1033 lr=3.00e-06 toks/s=427,227 tflops=115.01 mfu=11.63% step_ms=306.63 sm_util=95.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=40 loss=3.3677 pred=0.4568 aux=145.5464 lr=4.00e-06 toks/s=426,994 tflops=114.95 mfu=11.62% step_ms=306.81 sm_util=80.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=50 loss=3.3170 pred=0.4027 aux=145.7133 lr=5.00e-06 toks/s=452,584 tflops=121.84 mfu=12.32% step_ms=289.48 sm_util=67.0% hbm_util=44.0% mem_ctrl_util=44.0%
step=60 loss=3.2249 pred=0.3389 aux=144.2964 lr=6.00e-06 toks/s=449,988 tflops=121.14 mfu=12.25% step_ms=291.15 sm_util=83.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=70 loss=3.2072 pred=0.3214 aux=144.2927 lr=7.00e-06 toks/s=452,239 tflops=121.75 mfu=12.31% step_ms=289.62 sm_util=70.0% hbm_util=45.0% mem_ctrl_util=45.0%
step=80 loss=3.2055 pred=0.3231 aux=144.1190 lr=8.00e-06 toks/s=469,466 tflops=126.38 mfu=12.78% step_ms=279.06 sm_util=74.0% hbm_util=49.0% mem_ctrl_util=49.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=50.78
```
## per_expert_tile_1_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8932 pred=0.5797 aux=165.6708 lr=1.00e-06 toks/s=89,766 tflops=24.17 mfu=2.44% step_ms=1288.21 sm_util=32.0% hbm_util=23.0% mem_ctrl_util=23.0%
step=20 loss=3.7229 pred=0.5525 aux=158.5216 lr=2.00e-06 toks/s=235,444 tflops=63.38 mfu=6.41% step_ms=556.36 sm_util=39.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=30 loss=3.5807 pred=0.5190 aux=153.0894 lr=3.00e-06 toks/s=240,518 tflops=64.75 mfu=6.55% step_ms=544.78 sm_util=48.0% hbm_util=34.0% mem_ctrl_util=34.0%
step=40 loss=3.3675 pred=0.4567 aux=145.5408 lr=4.00e-06 toks/s=233,290 tflops=62.80 mfu=6.35% step_ms=561.71 sm_util=39.0% hbm_util=27.0% mem_ctrl_util=27.0%
step=50 loss=3.3171 pred=0.4026 aux=145.7231 lr=5.00e-06 toks/s=238,085 tflops=64.09 mfu=6.48% step_ms=550.39 sm_util=52.0% hbm_util=35.0% mem_ctrl_util=35.0%
step=60 loss=3.2248 pred=0.3389 aux=144.2966 lr=6.00e-06 toks/s=240,670 tflops=64.79 mfu=6.55% step_ms=544.48 sm_util=49.0% hbm_util=33.0% mem_ctrl_util=33.0%
step=70 loss=3.2070 pred=0.3213 aux=144.2851 lr=7.00e-06 toks/s=241,087 tflops=64.90 mfu=6.56% step_ms=543.55 sm_util=37.0% hbm_util=26.0% mem_ctrl_util=26.0%
step=80 loss=3.2051 pred=0.3230 aux=144.1054 lr=8.00e-06 toks/s=243,009 tflops=65.42 mfu=6.61% step_ms=539.25 sm_util=41.0% hbm_util=28.0% mem_ctrl_util=28.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=50.81
```
## per_expert_tile_64_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8942 pred=0.5797 aux=165.7243 lr=1.00e-06 toks/s=96,161 tflops=25.89 mfu=2.62% step_ms=1192.41 sm_util=51.0% hbm_util=36.0% mem_ctrl_util=36.0%
step=20 loss=3.7232 pred=0.5525 aux=158.5334 lr=2.00e-06 toks/s=351,201 tflops=94.55 mfu=9.56% step_ms=372.88 sm_util=63.0% hbm_util=42.0% mem_ctrl_util=42.0%
step=30 loss=3.5808 pred=0.5189 aux=153.0920 lr=3.00e-06 toks/s=387,493 tflops=104.32 mfu=10.55% step_ms=338.09 sm_util=62.0% hbm_util=40.0% mem_ctrl_util=40.0%
step=40 loss=3.3677 pred=0.4568 aux=145.5467 lr=4.00e-06 toks/s=408,599 tflops=110.00 mfu=11.12% step_ms=320.60 sm_util=69.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=50 loss=3.3167 pred=0.4027 aux=145.6996 lr=5.00e-06 toks/s=420,488 tflops=113.20 mfu=11.45% step_ms=311.59 sm_util=78.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=60 loss=3.2249 pred=0.3389 aux=144.2990 lr=6.00e-06 toks/s=428,184 tflops=115.27 mfu=11.66% step_ms=305.99 sm_util=64.0% hbm_util=41.0% mem_ctrl_util=41.0%
step=70 loss=3.2072 pred=0.3213 aux=144.2909 lr=7.00e-06 toks/s=432,827 tflops=116.52 mfu=11.78% step_ms=302.63 sm_util=80.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=80 loss=3.2055 pred=0.3230 aux=144.1259 lr=8.00e-06 toks/s=453,473 tflops=122.08 mfu=12.34% step_ms=288.92 sm_util=69.0% hbm_util=46.0% mem_ctrl_util=46.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=50.78
```
## scatter_compile_cap_0_9_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5738 pred=0.5873 aux=149.3209 lr=1.00e-06 toks/s=130,840 tflops=66.93 mfu=6.77% step_ms=830.28 sm_util=96.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=20 loss=3.4532 pred=0.5049 aux=147.4130 lr=2.00e-06 toks/s=337,659 tflops=172.72 mfu=17.46% step_ms=387.63 sm_util=0.0% hbm_util=0.0% mem_ctrl_util=0.0%
step=30 loss=3.4069 pred=0.4992 aux=145.3868 lr=3.00e-06 toks/s=460,353 tflops=235.48 mfu=23.81% step_ms=284.19 sm_util=85.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=40 loss=3.3493 pred=0.4654 aux=144.1953 lr=4.00e-06 toks/s=464,793 tflops=237.75 mfu=24.04% step_ms=281.52 sm_util=87.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=50 loss=3.2675 pred=0.3843 aux=144.1586 lr=5.00e-06 toks/s=474,326 tflops=242.63 mfu=24.53% step_ms=275.82 sm_util=82.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=3.2565 pred=0.3623 aux=144.7107 lr=6.00e-06 toks/s=470,897 tflops=240.87 mfu=24.36% step_ms=277.94 sm_util=78.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=70 loss=3.2467 pred=0.3526 aux=144.7040 lr=7.00e-06 toks/s=460,678 tflops=235.65 mfu=23.83% step_ms=283.96 sm_util=89.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=80 loss=3.2517 pred=0.3414 aux=145.5171 lr=8.00e-06 toks/s=462,464 tflops=236.56 mfu=23.92% step_ms=282.92 sm_util=90.0% hbm_util=61.0% mem_ctrl_util=61.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=44.58
```
## scatter_compile_cap_1_1_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5646 pred=0.5873 aux=148.8672 lr=1.00e-06 toks/s=129,432 tflops=69.50 mfu=7.03% step_ms=842.75 sm_util=96.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=3.4462 pred=0.5047 aux=147.0758 lr=2.00e-06 toks/s=327,734 tflops=175.99 mfu=17.80% step_ms=399.40 sm_util=34.0% hbm_util=22.0% mem_ctrl_util=22.0%
step=30 loss=3.4035 pred=0.4988 aux=145.2382 lr=3.00e-06 toks/s=442,612 tflops=237.68 mfu=24.03% step_ms=295.66 sm_util=85.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=40 loss=3.3484 pred=0.4648 aux=144.1768 lr=4.00e-06 toks/s=446,893 tflops=239.98 mfu=24.27% step_ms=292.82 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=50 loss=3.2665 pred=0.3839 aux=144.1264 lr=5.00e-06 toks/s=454,854 tflops=244.26 mfu=24.70% step_ms=287.66 sm_util=79.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.2535 pred=0.3619 aux=144.5813 lr=6.00e-06 toks/s=453,326 tflops=243.44 mfu=24.61% step_ms=288.73 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=70 loss=3.2382 pred=0.3523 aux=144.2979 lr=7.00e-06 toks/s=443,194 tflops=238.00 mfu=24.06% step_ms=295.20 sm_util=94.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=80 loss=3.2561 pred=0.3406 aux=145.7734 lr=8.00e-06 toks/s=446,025 tflops=239.52 mfu=24.22% step_ms=293.37 sm_util=96.0% hbm_util=67.0% mem_ctrl_util=67.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=48.08
```
## scatter_compile_cap_1_5_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5516 pred=0.5873 aux=148.2176 lr=1.00e-06 toks/s=127,236 tflops=74.81 mfu=7.56% step_ms=859.21 sm_util=97.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=20 loss=3.4307 pred=0.5048 aux=146.2980 lr=2.00e-06 toks/s=310,661 tflops=182.66 mfu=18.47% step_ms=421.38 sm_util=94.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=30 loss=3.3971 pred=0.4986 aux=144.9211 lr=3.00e-06 toks/s=412,042 tflops=242.26 mfu=24.50% step_ms=317.58 sm_util=96.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=40 loss=3.3516 pred=0.4649 aux=144.3366 lr=4.00e-06 toks/s=416,409 tflops=244.83 mfu=24.76% step_ms=314.32 sm_util=78.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=50 loss=3.2658 pred=0.3841 aux=144.0827 lr=5.00e-06 toks/s=423,665 tflops=249.10 mfu=25.19% step_ms=308.87 sm_util=84.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=60 loss=3.2512 pred=0.3619 aux=144.4619 lr=6.00e-06 toks/s=421,991 tflops=248.11 mfu=25.09% step_ms=310.23 sm_util=79.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=70 loss=3.2398 pred=0.3519 aux=144.3960 lr=7.00e-06 toks/s=412,731 tflops=242.67 mfu=24.54% step_ms=317.01 sm_util=78.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=80 loss=3.2370 pred=0.3403 aux=144.8360 lr=8.00e-06 toks/s=414,306 tflops=243.60 mfu=24.63% step_ms=315.88 sm_util=79.0% hbm_util=54.0% mem_ctrl_util=54.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=55.09
```
## scatter_compile_tile_128_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5531 pred=0.5872 aux=148.2943 lr=1.00e-06 toks/s=137,746 tflops=77.48 mfu=7.83% step_ms=781.46 sm_util=94.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=20 loss=3.4388 pred=0.5046 aux=146.7107 lr=2.00e-06 toks/s=432,678 tflops=243.37 mfu=24.61% step_ms=302.41 sm_util=76.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=30 loss=3.3991 pred=0.4985 aux=145.0269 lr=3.00e-06 toks/s=428,951 tflops=241.28 mfu=24.40% step_ms=305.02 sm_util=85.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=40 loss=3.3504 pred=0.4646 aux=144.2857 lr=4.00e-06 toks/s=433,013 tflops=243.56 mfu=24.63% step_ms=302.22 sm_util=75.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=50 loss=3.2654 pred=0.3838 aux=144.0792 lr=5.00e-06 toks/s=440,452 tflops=247.75 mfu=25.05% step_ms=297.08 sm_util=75.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=60 loss=3.2510 pred=0.3618 aux=144.4602 lr=6.00e-06 toks/s=436,860 tflops=245.73 mfu=24.85% step_ms=299.64 sm_util=82.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=3.2377 pred=0.3519 aux=144.2916 lr=7.00e-06 toks/s=428,230 tflops=240.87 mfu=24.36% step_ms=305.52 sm_util=77.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=80 loss=3.2470 pred=0.3403 aux=145.3324 lr=8.00e-06 toks/s=430,133 tflops=241.94 mfu=24.46% step_ms=304.24 sm_util=96.0% hbm_util=68.0% mem_ctrl_util=68.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=51.79
```
## scatter_compile_tile_64_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5532 pred=0.5872 aux=148.2959 lr=1.00e-06 toks/s=138,037 tflops=77.64 mfu=7.85% step_ms=779.49 sm_util=96.0% hbm_util=69.0% mem_ctrl_util=69.0%
step=20 loss=3.4387 pred=0.5046 aux=146.7033 lr=2.00e-06 toks/s=430,830 tflops=242.33 mfu=24.50% step_ms=303.71 sm_util=80.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=30 loss=3.3991 pred=0.4985 aux=145.0275 lr=3.00e-06 toks/s=429,212 tflops=241.42 mfu=24.41% step_ms=304.83 sm_util=78.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=40 loss=3.3503 pred=0.4646 aux=144.2815 lr=4.00e-06 toks/s=432,772 tflops=243.43 mfu=24.61% step_ms=302.38 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=50 loss=3.2652 pred=0.3838 aux=144.0666 lr=5.00e-06 toks/s=440,565 tflops=247.81 mfu=25.06% step_ms=296.99 sm_util=78.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.2508 pred=0.3618 aux=144.4507 lr=6.00e-06 toks/s=437,379 tflops=246.02 mfu=24.88% step_ms=299.27 sm_util=76.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=70 loss=3.2374 pred=0.3519 aux=144.2751 lr=7.00e-06 toks/s=428,594 tflops=241.08 mfu=24.38% step_ms=305.25 sm_util=89.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=80 loss=3.2473 pred=0.3403 aux=145.3510 lr=8.00e-06 toks/s=431,353 tflops=242.63 mfu=24.53% step_ms=303.37 sm_util=89.0% hbm_util=62.0% mem_ctrl_util=62.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=51.79
```
