# Experiment Report

Generated from logs in `experiments/20260118_020534`.


## Summary


| Experiment | Params Total | Params Active | Precision | Toks/Sec | TFLOPS | MFU | Step ms |
|---|---|---|---|---|---|---|---|
| dense_ffn_bf16_flash_ | 39.00M | 39.00M | bf16 | 482,406 | 129.87 | 13.13 | 271.22 |
| dense_ffn_bf16_flash_compile_ | 39.00M | 39.00M | bf16 | 600,430 | 161.64 | 16.34 | 218.04 |
| dense_ffn_bf16_sdpa_ | 39.00M | 39.00M | bf16 | 247,595 | 66.65 | 6.74 | 527.25 |
| logical_dense_cap_0_9_ | 113.35M | 49.65M | bf16 | 399,769 | 204.49 | 20.68 | 327.08 |
| logical_dense_cap_1_1_ | 113.35M | 49.65M | bf16 | 384,590 | 206.52 | 20.88 | 340.04 |
| logical_dense_cap_1_3_ | 113.35M | 49.65M | bf16 | 371,179 | 208.78 | 21.11 | 352.35 |
| logical_dense_cap_1_5_ | 113.35M | 49.65M | bf16 | 357,860 | 210.41 | 21.27 | 365.49 |
| moe_naive_bf16_ | 113.35M | 113.35M | bf16 | 298,743 | 80.42 | 8.13 | 438.33 |
| moe_naive_bf16_compile_ | 113.35M | 113.35M | bf16 | 292,810 | 78.83 | 7.97 | 447.45 |
| moe_scatter_bf16_ | 113.35M | 49.65M | bf16 | 372,272 | 209.40 | 21.17 | 351.31 |
| moe_scatter_bf16_compile_ | 113.35M | 49.65M | bf16 | 481,297 | 270.72 | 27.37 | 271.81 |
| per_expert_tile_128_ | 113.35M | 113.35M | bf16 | 412,972 | 111.17 | 11.24 | 316.95 |
| per_expert_tile_1_ | 113.35M | 113.35M | bf16 | 293,730 | 79.07 | 8.00 | 445.80 |
| per_expert_tile_64_ | 113.35M | 113.35M | bf16 | 410,898 | 110.62 | 11.18 | 318.60 |
| scatter_compile_cap_0_9_ | 113.35M | 49.65M | bf16 | 513,440 | 262.64 | 26.56 | 254.76 |
| scatter_compile_cap_1_1_ | 113.35M | 49.65M | bf16 | 491,737 | 264.06 | 26.70 | 266.03 |
| scatter_compile_cap_1_5_ | 113.35M | 49.65M | bf16 | 464,151 | 272.90 | 27.59 | 281.83 |
| scatter_compile_tile_128_ | 113.35M | 49.65M | bf16 | 479,761 | 269.86 | 27.29 | 272.66 |
| scatter_compile_tile_64_ | 113.35M | 49.65M | bf16 | 480,331 | 270.18 | 27.32 | 272.34 |



## dense_ffn_bf16_flash_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.5331 pred=0.6531 aux=144.0000 lr=1.00e-06 toks/s=245,379 tflops=66.06 mfu=6.68% step_ms=409.37 sm_util=99.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=3.4422 pred=0.5622 aux=144.0000 lr=2.00e-06 toks/s=482,535 tflops=129.90 mfu=13.13% step_ms=271.15 sm_util=97.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=30 loss=3.3260 pred=0.4460 aux=144.0000 lr=3.00e-06 toks/s=480,473 tflops=129.35 mfu=13.08% step_ms=272.26 sm_util=94.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=3.3151 pred=0.4351 aux=144.0000 lr=4.00e-06 toks/s=487,454 tflops=131.23 mfu=13.27% step_ms=268.39 sm_util=97.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=50 loss=3.2280 pred=0.3480 aux=144.0000 lr=5.00e-06 toks/s=489,966 tflops=131.90 mfu=13.34% step_ms=266.90 sm_util=91.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.2051 pred=0.3251 aux=144.0000 lr=6.00e-06 toks/s=486,247 tflops=130.90 mfu=13.24% step_ms=269.06 sm_util=98.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=70 loss=3.1957 pred=0.3157 aux=144.0000 lr=7.00e-06 toks/s=499,919 tflops=134.58 mfu=13.61% step_ms=261.62 sm_util=99.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=80 loss=3.2114 pred=0.3314 aux=144.0000 lr=8.00e-06 toks/s=482,406 tflops=129.87 mfu=13.13% step_ms=271.22 sm_util=94.0% hbm_util=57.0% mem_ctrl_util=57.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=39.68
```
## dense_ffn_bf16_flash_compile_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
/home/ubuntu/py312/lib/python3.12/site-packages/torch/_inductor/lowering.py:7242: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
/home/ubuntu/py312/lib/python3.12/site-packages/torch/_inductor/lowering.py:7242: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
step=10 loss=3.5331 pred=0.6531 aux=144.0000 lr=1.00e-06 toks/s=70,617 tflops=19.01 mfu=1.92% step_ms=1709.07 sm_util=88.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=20 loss=3.4420 pred=0.5620 aux=144.0000 lr=2.00e-06 toks/s=596,488 tflops=160.58 mfu=16.24% step_ms=219.39 sm_util=82.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=30 loss=3.3258 pred=0.4458 aux=144.0000 lr=3.00e-06 toks/s=592,405 tflops=159.48 mfu=16.13% step_ms=220.96 sm_util=87.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=40 loss=3.3149 pred=0.4349 aux=144.0000 lr=4.00e-06 toks/s=606,413 tflops=163.25 mfu=16.51% step_ms=215.87 sm_util=78.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=50 loss=3.2278 pred=0.3478 aux=144.0000 lr=5.00e-06 toks/s=608,906 tflops=163.92 mfu=16.57% step_ms=214.87 sm_util=88.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=3.2049 pred=0.3249 aux=144.0000 lr=6.00e-06 toks/s=602,956 tflops=162.32 mfu=16.41% step_ms=217.11 sm_util=82.0% hbm_util=54.0% mem_ctrl_util=54.0%
step=70 loss=3.1956 pred=0.3156 aux=144.0000 lr=7.00e-06 toks/s=624,159 tflops=168.03 mfu=16.99% step_ms=209.66 sm_util=89.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=80 loss=3.2114 pred=0.3314 aux=144.0000 lr=8.00e-06 toks/s=600,430 tflops=161.64 mfu=16.34% step_ms=218.04 sm_util=92.0% hbm_util=60.0% mem_ctrl_util=60.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=32.64
```
## dense_ffn_bf16_sdpa_

```
params total=39.00M (39,001,833) active=39.00M (39,001,833)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=nan pred=nan aux=nan lr=1.00e-06 toks/s=165,129 tflops=44.45 mfu=4.49% step_ms=668.86 sm_util=100.0% hbm_util=31.0% mem_ctrl_util=31.0%
step=20 loss=nan pred=nan aux=nan lr=2.00e-06 toks/s=247,359 tflops=66.59 mfu=6.73% step_ms=527.76 sm_util=98.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=30 loss=nan pred=nan aux=nan lr=3.00e-06 toks/s=247,312 tflops=66.58 mfu=6.73% step_ms=527.86 sm_util=100.0% hbm_util=29.0% mem_ctrl_util=29.0%
step=40 loss=nan pred=nan aux=nan lr=4.00e-06 toks/s=247,978 tflops=66.76 mfu=6.75% step_ms=526.51 sm_util=100.0% hbm_util=32.0% mem_ctrl_util=32.0%
step=50 loss=nan pred=nan aux=nan lr=5.00e-06 toks/s=248,428 tflops=66.88 mfu=6.76% step_ms=525.48 sm_util=100.0% hbm_util=33.0% mem_ctrl_util=33.0%
step=60 loss=nan pred=nan aux=nan lr=6.00e-06 toks/s=248,258 tflops=66.83 mfu=6.76% step_ms=525.86 sm_util=100.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=70 loss=nan pred=nan aux=nan lr=7.00e-06 toks/s=249,360 tflops=67.13 mfu=6.79% step_ms=523.50 sm_util=99.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=80 loss=nan pred=nan aux=nan lr=8.00e-06 toks/s=247,595 tflops=66.65 mfu=6.74% step_ms=527.25 sm_util=100.0% hbm_util=32.0% mem_ctrl_util=32.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=45.93
```
## logical_dense_cap_0_9_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2022.6055 pred=2019.6866 aux=145.9396 lr=1.00e-06 toks/s=212,341 tflops=108.62 mfu=10.98% step_ms=488.18 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0%
step=20 loss=2080.5332 pred=2077.6250 aux=145.4065 lr=2.00e-06 toks/s=393,757 tflops=201.42 mfu=20.37% step_ms=332.19 sm_util=95.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=30 loss=2066.8667 pred=2063.9529 aux=145.6916 lr=3.00e-06 toks/s=399,407 tflops=204.31 mfu=20.66% step_ms=327.28 sm_util=94.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=40 loss=1999.7657 pred=1996.8483 aux=145.8739 lr=4.00e-06 toks/s=392,246 tflops=200.64 mfu=20.29% step_ms=333.29 sm_util=97.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=50 loss=2045.8881 pred=2042.9603 aux=146.3853 lr=5.00e-06 toks/s=392,946 tflops=201.00 mfu=20.32% step_ms=332.79 sm_util=94.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=60 loss=2037.2145 pred=2034.2897 aux=146.2397 lr=6.00e-06 toks/s=391,096 tflops=200.05 mfu=20.23% step_ms=334.28 sm_util=99.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=70 loss=2040.4167 pred=2037.4838 aux=146.6473 lr=7.00e-06 toks/s=393,806 tflops=201.44 mfu=20.37% step_ms=331.94 sm_util=94.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=1973.8569 pred=1970.9390 aux=145.8978 lr=8.00e-06 toks/s=399,769 tflops=204.49 mfu=20.68% step_ms=327.08 sm_util=99.0% hbm_util=69.0% mem_ctrl_util=69.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=53.84
```
## logical_dense_cap_1_1_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2070.1821 pred=2067.2544 aux=146.3857 lr=1.00e-06 toks/s=209,171 tflops=112.33 mfu=11.36% step_ms=493.84 sm_util=98.0% hbm_util=75.0% mem_ctrl_util=75.0%
step=20 loss=2076.2419 pred=2073.3267 aux=145.7585 lr=2.00e-06 toks/s=379,025 tflops=203.54 mfu=20.58% step_ms=345.21 sm_util=95.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=30 loss=2098.2407 pred=2095.3228 aux=145.8990 lr=3.00e-06 toks/s=384,033 tflops=206.23 mfu=20.85% step_ms=340.41 sm_util=94.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=2013.5730 pred=2010.6544 aux=145.9312 lr=4.00e-06 toks/s=377,146 tflops=202.53 mfu=20.48% step_ms=346.66 sm_util=94.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=50 loss=2090.0896 pred=2087.1694 aux=146.0135 lr=5.00e-06 toks/s=377,559 tflops=202.75 mfu=20.50% step_ms=346.39 sm_util=99.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=60 loss=2046.8396 pred=2043.9167 aux=146.1418 lr=6.00e-06 toks/s=376,095 tflops=201.96 mfu=20.42% step_ms=347.66 sm_util=93.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=70 loss=2021.9105 pred=2018.9879 aux=146.1306 lr=7.00e-06 toks/s=378,329 tflops=203.16 mfu=20.54% step_ms=345.55 sm_util=99.0% hbm_util=71.0% mem_ctrl_util=71.0%
step=80 loss=2002.6135 pred=1999.6936 aux=145.9952 lr=8.00e-06 toks/s=384,590 tflops=206.52 mfu=20.88% step_ms=340.04 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=58.38
```
## logical_dense_cap_1_3_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2067.7847 pred=2064.8611 aux=146.1801 lr=1.00e-06 toks/s=205,108 tflops=115.37 mfu=11.67% step_ms=509.09 sm_util=98.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=20 loss=2103.7097 pred=2100.7905 aux=145.9569 lr=2.00e-06 toks/s=366,239 tflops=206.00 mfu=20.83% step_ms=357.19 sm_util=99.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=30 loss=2095.2642 pred=2092.3486 aux=145.7745 lr=3.00e-06 toks/s=371,289 tflops=208.84 mfu=21.12% step_ms=352.13 sm_util=93.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=2032.5171 pred=2029.6028 aux=145.7142 lr=4.00e-06 toks/s=365,043 tflops=205.33 mfu=20.76% step_ms=358.20 sm_util=92.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=50 loss=2087.2708 pred=2084.3564 aux=145.7199 lr=5.00e-06 toks/s=365,145 tflops=205.39 mfu=20.77% step_ms=358.17 sm_util=92.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=60 loss=2077.9082 pred=2074.9917 aux=145.8259 lr=6.00e-06 toks/s=364,179 tflops=204.84 mfu=20.71% step_ms=359.07 sm_util=93.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=70 loss=2027.1243 pred=2024.2087 aux=145.7762 lr=7.00e-06 toks/s=365,821 tflops=205.77 mfu=20.81% step_ms=357.44 sm_util=93.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=80 loss=1987.1306 pred=1984.2100 aux=146.0356 lr=8.00e-06 toks/s=371,179 tflops=208.78 mfu=21.11% step_ms=352.35 sm_util=96.0% hbm_util=70.0% mem_ctrl_util=70.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=62.87
```
## logical_dense_cap_1_5_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2062.4753 pred=2059.5579 aux=145.8736 lr=1.00e-06 toks/s=198,050 tflops=116.45 mfu=11.77% step_ms=525.81 sm_util=99.0% hbm_util=77.0% mem_ctrl_util=77.0%
step=20 loss=2110.1323 pred=2107.2148 aux=145.8772 lr=2.00e-06 toks/s=353,417 tflops=207.79 mfu=21.01% step_ms=370.16 sm_util=95.0% hbm_util=65.0% mem_ctrl_util=65.0%
step=30 loss=2106.9624 pred=2104.0479 aux=145.7298 lr=3.00e-06 toks/s=358,099 tflops=210.55 mfu=21.29% step_ms=365.13 sm_util=99.0% hbm_util=74.0% mem_ctrl_util=74.0%
step=40 loss=2055.2568 pred=2052.3298 aux=146.3515 lr=4.00e-06 toks/s=351,906 tflops=206.91 mfu=20.92% step_ms=371.60 sm_util=93.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=50 loss=2080.9839 pred=2078.0632 aux=146.0340 lr=5.00e-06 toks/s=352,643 tflops=207.34 mfu=20.96% step_ms=370.91 sm_util=99.0% hbm_util=72.0% mem_ctrl_util=72.0%
step=60 loss=2083.7253 pred=2080.8013 aux=146.2009 lr=6.00e-06 toks/s=351,154 tflops=206.46 mfu=20.88% step_ms=372.42 sm_util=93.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=70 loss=2045.6603 pred=2042.7327 aux=146.3831 lr=7.00e-06 toks/s=353,352 tflops=207.76 mfu=21.01% step_ms=370.04 sm_util=99.0% hbm_util=75.0% mem_ctrl_util=75.0%
step=80 loss=2008.0089 pred=2005.0823 aux=146.3297 lr=8.00e-06 toks/s=357,860 tflops=210.41 mfu=21.27% step_ms=365.49 sm_util=92.0% hbm_util=63.0% mem_ctrl_util=63.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=66.71
```
## moe_naive_bf16_

```
params total=113.35M (113,351,913) active=113.35M (113,351,913)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6192 pred=0.5798 aux=151.9703 lr=1.00e-06 toks/s=176,224 tflops=47.44 mfu=4.80% step_ms=596.30 sm_util=64.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=20 loss=3.5134 pred=0.5526 aux=148.0434 lr=2.00e-06 toks/s=286,373 tflops=77.09 mfu=7.80% step_ms=457.26 sm_util=69.0% hbm_util=47.0% mem_ctrl_util=47.0%
step=30 loss=3.3970 pred=0.5191 aux=143.8947 lr=3.00e-06 toks/s=287,477 tflops=77.39 mfu=7.83% step_ms=455.43 sm_util=58.0% hbm_util=37.0% mem_ctrl_util=37.0%
step=40 loss=3.3153 pred=0.4569 aux=142.9212 lr=4.00e-06 toks/s=284,624 tflops=76.62 mfu=7.75% step_ms=460.05 sm_util=64.0% hbm_util=41.0% mem_ctrl_util=41.0%
step=50 loss=3.3167 pred=0.4029 aux=145.6935 lr=5.00e-06 toks/s=294,189 tflops=79.20 mfu=8.01% step_ms=445.15 sm_util=83.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=3.2236 pred=0.3390 aux=144.2274 lr=6.00e-06 toks/s=297,155 tflops=80.00 mfu=8.09% step_ms=440.65 sm_util=71.0% hbm_util=48.0% mem_ctrl_util=48.0%
step=70 loss=3.1920 pred=0.3214 aux=143.5274 lr=7.00e-06 toks/s=296,477 tflops=79.81 mfu=8.07% step_ms=441.62 sm_util=69.0% hbm_util=47.0% mem_ctrl_util=47.0%
step=80 loss=3.2029 pred=0.3231 aux=143.9904 lr=8.00e-06 toks/s=298,743 tflops=80.42 mfu=8.13% step_ms=438.33 sm_util=79.0% hbm_util=54.0% mem_ctrl_util=54.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.90
```
## moe_naive_bf16_compile_

```
params total=113.35M (113,351,913) active=113.35M (113,351,913)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.8932 pred=0.5797 aux=165.6746 lr=1.00e-06 toks/s=79,400 tflops=21.38 mfu=2.16% step_ms=1501.87 sm_util=41.0% hbm_util=29.0% mem_ctrl_util=29.0%
step=20 loss=3.7228 pred=0.5525 aux=158.5183 lr=2.00e-06 toks/s=283,108 tflops=76.21 mfu=7.71% step_ms=462.65 sm_util=56.0% hbm_util=38.0% mem_ctrl_util=38.0%
step=30 loss=3.5805 pred=0.5189 aux=153.0777 lr=3.00e-06 toks/s=287,806 tflops=77.48 mfu=7.83% step_ms=455.16 sm_util=56.0% hbm_util=39.0% mem_ctrl_util=39.0%
step=40 loss=3.3674 pred=0.4567 aux=145.5390 lr=4.00e-06 toks/s=282,002 tflops=75.92 mfu=7.68% step_ms=464.56 sm_util=41.0% hbm_util=28.0% mem_ctrl_util=28.0%
step=50 loss=3.3169 pred=0.4026 aux=145.7148 lr=5.00e-06 toks/s=286,698 tflops=77.18 mfu=7.80% step_ms=457.00 sm_util=59.0% hbm_util=40.0% mem_ctrl_util=40.0%
step=60 loss=3.2250 pred=0.3389 aux=144.3041 lr=6.00e-06 toks/s=292,134 tflops=78.64 mfu=7.95% step_ms=448.47 sm_util=61.0% hbm_util=42.0% mem_ctrl_util=42.0%
step=70 loss=3.2070 pred=0.3213 aux=144.2863 lr=7.00e-06 toks/s=295,582 tflops=79.57 mfu=8.05% step_ms=443.20 sm_util=48.0% hbm_util=32.0% mem_ctrl_util=32.0%
step=80 loss=3.2051 pred=0.3229 aux=144.1083 lr=8.00e-06 toks/s=292,810 tflops=78.83 mfu=7.97% step_ms=447.45 sm_util=57.0% hbm_util=40.0% mem_ctrl_util=40.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=50.81
```
## moe_scatter_bf16_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2065.4121 pred=2062.4888 aux=146.1697 lr=1.00e-06 toks/s=205,780 tflops=115.75 mfu=11.70% step_ms=508.53 sm_util=99.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=20 loss=2102.5220 pred=2099.6030 aux=145.9502 lr=2.00e-06 toks/s=366,943 tflops=206.40 mfu=20.87% step_ms=356.49 sm_util=99.0% hbm_util=70.0% mem_ctrl_util=70.0%
step=30 loss=2105.1399 pred=2102.2231 aux=145.8374 lr=3.00e-06 toks/s=371,879 tflops=209.17 mfu=21.15% step_ms=351.59 sm_util=94.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=2033.0956 pred=2030.1785 aux=145.8582 lr=4.00e-06 toks/s=365,569 tflops=205.63 mfu=20.79% step_ms=357.67 sm_util=92.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=50 loss=2101.3428 pred=2098.4192 aux=146.1791 lr=5.00e-06 toks/s=366,203 tflops=205.98 mfu=20.83% step_ms=357.13 sm_util=93.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=60 loss=2067.7375 pred=2064.8098 aux=146.3861 lr=6.00e-06 toks/s=364,772 tflops=205.18 mfu=20.75% step_ms=358.48 sm_util=94.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=70 loss=2030.9683 pred=2028.0410 aux=146.3621 lr=7.00e-06 toks/s=366,992 tflops=206.43 mfu=20.87% step_ms=356.38 sm_util=94.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=80 loss=1983.1749 pred=1980.2509 aux=146.2064 lr=8.00e-06 toks/s=372,272 tflops=209.40 mfu=21.17% step_ms=351.31 sm_util=96.0% hbm_util=63.0% mem_ctrl_util=63.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=62.38
```
## moe_scatter_bf16_compile_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2054.0859 pred=2050.8752 aux=160.5330 lr=1.00e-06 toks/s=75,829 tflops=42.65 mfu=4.31% step_ms=1578.63 sm_util=87.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=20 loss=2103.4226 pred=2100.3120 aux=155.5245 lr=2.00e-06 toks/s=471,540 tflops=265.23 mfu=26.82% step_ms=277.47 sm_util=96.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=30 loss=2093.8140 pred=2090.7305 aux=154.1769 lr=3.00e-06 toks/s=478,469 tflops=269.13 mfu=27.21% step_ms=273.33 sm_util=84.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=2042.4352 pred=2039.2183 aux=160.8485 lr=4.00e-06 toks/s=469,819 tflops=264.26 mfu=26.72% step_ms=278.37 sm_util=85.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=50 loss=2077.4697 pred=2074.3831 aux=154.3332 lr=5.00e-06 toks/s=469,626 tflops=264.16 mfu=26.71% step_ms=278.53 sm_util=83.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=2062.6377 pred=2059.5452 aux=154.6243 lr=6.00e-06 toks/s=468,779 tflops=263.68 mfu=26.66% step_ms=279.01 sm_util=81.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=70 loss=2037.0859 pred=2033.9305 aux=157.7726 lr=7.00e-06 toks/s=472,920 tflops=266.01 mfu=26.90% step_ms=276.52 sm_util=79.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=80 loss=1991.5975 pred=1988.3469 aux=162.5323 lr=8.00e-06 toks/s=481,297 tflops=270.72 mfu=27.37% step_ms=271.81 sm_util=91.0% hbm_util=66.0% mem_ctrl_util=66.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=52.35
```
## per_expert_tile_128_

```
params total=113.35M (113,351,913) active=113.35M (113,351,913)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6194 pred=0.5798 aux=151.9784 lr=1.00e-06 toks/s=213,081 tflops=57.36 mfu=5.80% step_ms=469.35 sm_util=90.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=3.5135 pred=0.5526 aux=148.0474 lr=2.00e-06 toks/s=388,967 tflops=104.71 mfu=10.59% step_ms=336.53 sm_util=100.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=30 loss=3.3969 pred=0.5191 aux=143.8904 lr=3.00e-06 toks/s=403,207 tflops=108.55 mfu=10.98% step_ms=324.57 sm_util=90.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=40 loss=3.3154 pred=0.4570 aux=142.9193 lr=4.00e-06 toks/s=394,000 tflops=106.07 mfu=10.72% step_ms=332.21 sm_util=94.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=50 loss=3.3173 pred=0.4029 aux=145.7188 lr=5.00e-06 toks/s=399,592 tflops=107.57 mfu=10.88% step_ms=327.54 sm_util=92.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=60 loss=3.2237 pred=0.3391 aux=144.2316 lr=6.00e-06 toks/s=398,689 tflops=107.33 mfu=10.85% step_ms=328.27 sm_util=89.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=70 loss=3.1917 pred=0.3215 aux=143.5068 lr=7.00e-06 toks/s=402,331 tflops=108.31 mfu=10.95% step_ms=325.27 sm_util=97.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=80 loss=3.2034 pred=0.3232 aux=144.0060 lr=8.00e-06 toks/s=412,972 tflops=111.17 mfu=11.24% step_ms=316.95 sm_util=95.0% hbm_util=62.0% mem_ctrl_util=62.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.88
```
## per_expert_tile_1_

```
params total=113.35M (113,351,913) active=113.35M (113,351,913)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6192 pred=0.5798 aux=151.9684 lr=1.00e-06 toks/s=176,878 tflops=47.62 mfu=4.81% step_ms=594.40 sm_util=54.0% hbm_util=38.0% mem_ctrl_util=38.0%
step=20 loss=3.5134 pred=0.5526 aux=148.0422 lr=2.00e-06 toks/s=284,637 tflops=76.63 mfu=7.75% step_ms=460.05 sm_util=79.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=30 loss=3.3970 pred=0.5191 aux=143.8934 lr=3.00e-06 toks/s=281,308 tflops=75.73 mfu=7.66% step_ms=465.41 sm_util=67.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=40 loss=3.3153 pred=0.4569 aux=142.9192 lr=4.00e-06 toks/s=282,553 tflops=76.07 mfu=7.69% step_ms=463.41 sm_util=62.0% hbm_util=42.0% mem_ctrl_util=42.0%
step=50 loss=3.3169 pred=0.4029 aux=145.7008 lr=5.00e-06 toks/s=292,504 tflops=78.74 mfu=7.96% step_ms=447.67 sm_util=67.0% hbm_util=46.0% mem_ctrl_util=46.0%
step=60 loss=3.2236 pred=0.3390 aux=144.2297 lr=6.00e-06 toks/s=294,678 tflops=79.33 mfu=8.02% step_ms=444.36 sm_util=73.0% hbm_util=50.0% mem_ctrl_util=50.0%
step=70 loss=3.1916 pred=0.3214 aux=143.5111 lr=7.00e-06 toks/s=297,347 tflops=80.05 mfu=8.09% step_ms=440.31 sm_util=63.0% hbm_util=44.0% mem_ctrl_util=44.0%
step=80 loss=3.2032 pred=0.3231 aux=144.0060 lr=8.00e-06 toks/s=293,730 tflops=79.07 mfu=8.00% step_ms=445.80 sm_util=65.0% hbm_util=46.0% mem_ctrl_util=46.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.90
```
## per_expert_tile_64_

```
params total=113.35M (113,351,913) active=113.35M (113,351,913)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=3.6194 pred=0.5798 aux=151.9798 lr=1.00e-06 toks/s=201,039 tflops=54.12 mfu=5.47% step_ms=500.58 sm_util=82.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=20 loss=3.5134 pred=0.5526 aux=148.0429 lr=2.00e-06 toks/s=372,499 tflops=100.28 mfu=10.14% step_ms=351.43 sm_util=85.0% hbm_util=53.0% mem_ctrl_util=53.0%
step=30 loss=3.3970 pred=0.5191 aux=143.8925 lr=3.00e-06 toks/s=388,081 tflops=104.47 mfu=10.56% step_ms=337.24 sm_util=88.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=40 loss=3.3153 pred=0.4569 aux=142.9176 lr=4.00e-06 toks/s=383,606 tflops=103.27 mfu=10.44% step_ms=341.20 sm_util=83.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=50 loss=3.3171 pred=0.4029 aux=145.7104 lr=5.00e-06 toks/s=392,251 tflops=105.60 mfu=10.68% step_ms=333.71 sm_util=95.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=60 loss=3.2237 pred=0.3390 aux=144.2317 lr=6.00e-06 toks/s=395,418 tflops=106.45 mfu=10.76% step_ms=331.00 sm_util=86.0% hbm_util=51.0% mem_ctrl_util=51.0%
step=70 loss=3.1918 pred=0.3215 aux=143.5191 lr=7.00e-06 toks/s=398,220 tflops=107.20 mfu=10.84% step_ms=328.60 sm_util=94.0% hbm_util=63.0% mem_ctrl_util=63.0%
step=80 loss=3.2030 pred=0.3232 aux=143.9938 lr=8.00e-06 toks/s=410,898 tflops=110.62 mfu=11.18% step_ms=318.60 sm_util=95.0% hbm_util=63.0% mem_ctrl_util=63.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=59.88
```
## scatter_compile_cap_0_9_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2017.1937 pred=2013.9666 aux=161.3615 lr=1.00e-06 toks/s=94,444 tflops=48.31 mfu=4.88% step_ms=1239.02 sm_util=87.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=20 loss=2074.8311 pred=2071.6846 aux=157.3267 lr=2.00e-06 toks/s=499,474 tflops=255.49 mfu=25.83% step_ms=261.92 sm_util=88.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=30 loss=2052.4905 pred=2049.3489 aux=157.0816 lr=3.00e-06 toks/s=362,414 tflops=185.38 mfu=18.74% step_ms=361.02 sm_util=86.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=2022.2773 pred=2018.9691 aux=165.4111 lr=4.00e-06 toks/s=499,908 tflops=255.71 mfu=25.86% step_ms=261.61 sm_util=77.0% hbm_util=52.0% mem_ctrl_util=52.0%
step=50 loss=2040.0808 pred=2036.8833 aux=159.8769 lr=5.00e-06 toks/s=498,886 tflops=255.19 mfu=25.80% step_ms=262.16 sm_util=81.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=60 loss=2027.3199 pred=2024.1321 aux=159.3949 lr=6.00e-06 toks/s=499,397 tflops=255.45 mfu=25.83% step_ms=261.87 sm_util=84.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=70 loss=2013.7950 pred=2010.5715 aux=161.1773 lr=7.00e-06 toks/s=501,836 tflops=256.70 mfu=25.96% step_ms=260.53 sm_util=88.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=80 loss=1990.7428 pred=1987.4607 aux=164.1069 lr=8.00e-06 toks/s=513,440 tflops=262.64 mfu=26.56% step_ms=254.76 sm_util=79.0% hbm_util=57.0% mem_ctrl_util=57.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=45.54
```
## scatter_compile_cap_1_1_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2051.6829 pred=2048.4590 aux=161.1937 lr=1.00e-06 toks/s=93,393 tflops=50.15 mfu=5.07% step_ms=1252.69 sm_util=84.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=20 loss=2080.8240 pred=2077.7048 aux=155.9551 lr=2.00e-06 toks/s=483,038 tflops=259.39 mfu=26.23% step_ms=270.85 sm_util=85.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=30 loss=2085.8618 pred=2082.7441 aux=155.8794 lr=3.00e-06 toks/s=354,967 tflops=190.62 mfu=19.27% step_ms=368.61 sm_util=89.0% hbm_util=61.0% mem_ctrl_util=61.0%
step=40 loss=2037.7424 pred=2034.4736 aux=163.4421 lr=4.00e-06 toks/s=481,900 tflops=258.78 mfu=26.17% step_ms=271.39 sm_util=91.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=50 loss=2073.9424 pred=2070.8215 aux=156.0417 lr=5.00e-06 toks/s=481,321 tflops=258.47 mfu=26.13% step_ms=271.75 sm_util=80.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=60 loss=2040.4156 pred=2037.2766 aux=156.9546 lr=6.00e-06 toks/s=480,768 tflops=258.17 mfu=26.10% step_ms=272.04 sm_util=94.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=70 loss=2027.4744 pred=2024.2754 aux=159.9467 lr=7.00e-06 toks/s=483,701 tflops=259.75 mfu=26.26% step_ms=270.35 sm_util=84.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=80 loss=1987.6449 pred=1984.3933 aux=162.5764 lr=8.00e-06 toks/s=491,737 tflops=264.06 mfu=26.70% step_ms=266.03 sm_util=96.0% hbm_util=66.0% mem_ctrl_util=66.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=48.47
```
## scatter_compile_cap_1_5_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2055.3096 pred=2052.1445 aux=158.2539 lr=1.00e-06 toks/s=95,047 tflops=55.88 mfu=5.65% step_ms=1232.01 sm_util=97.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=20 loss=2108.0684 pred=2104.9810 aux=154.3695 lr=2.00e-06 toks/s=456,048 tflops=268.14 mfu=27.11% step_ms=286.92 sm_util=81.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=30 loss=2106.0334 pred=2102.9668 aux=153.3380 lr=3.00e-06 toks/s=340,766 tflops=200.36 mfu=20.26% step_ms=383.99 sm_util=89.0% hbm_util=67.0% mem_ctrl_util=67.0%
step=40 loss=2057.1311 pred=2053.9526 aux=158.9202 lr=4.00e-06 toks/s=454,586 tflops=267.28 mfu=27.03% step_ms=287.73 sm_util=86.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=50 loss=2093.1646 pred=2090.0779 aux=154.3360 lr=5.00e-06 toks/s=453,250 tflops=266.49 mfu=26.95% step_ms=288.60 sm_util=86.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=60 loss=2090.1729 pred=2087.0923 aux=154.0272 lr=6.00e-06 toks/s=452,525 tflops=266.07 mfu=26.90% step_ms=289.05 sm_util=90.0% hbm_util=64.0% mem_ctrl_util=64.0%
step=70 loss=2038.0928 pred=2034.9254 aux=158.3706 lr=7.00e-06 toks/s=455,398 tflops=267.76 mfu=27.07% step_ms=287.17 sm_util=83.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=80 loss=1992.4608 pred=1989.2382 aux=161.1306 lr=8.00e-06 toks/s=464,151 tflops=272.90 mfu=27.59% step_ms=281.83 sm_util=88.0% hbm_util=64.0% mem_ctrl_util=64.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=56.06
```
## scatter_compile_tile_128_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2055.9841 pred=2052.7729 aux=160.5591 lr=1.00e-06 toks/s=124,709 tflops=70.15 mfu=7.09% step_ms=903.17 sm_util=95.0% hbm_util=73.0% mem_ctrl_util=73.0%
step=20 loss=2101.6804 pred=2098.5686 aux=155.5961 lr=2.00e-06 toks/s=471,958 tflops=265.47 mfu=26.84% step_ms=277.23 sm_util=98.0% hbm_util=68.0% mem_ctrl_util=68.0%
step=30 loss=2092.7253 pred=2089.6409 aux=154.2283 lr=3.00e-06 toks/s=479,821 tflops=269.89 mfu=27.29% step_ms=272.53 sm_util=80.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=40 loss=2020.9282 pred=2017.7200 aux=160.4138 lr=4.00e-06 toks/s=469,305 tflops=263.98 mfu=26.69% step_ms=278.66 sm_util=79.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=50 loss=2079.0496 pred=2075.9443 aux=155.2575 lr=5.00e-06 toks/s=468,961 tflops=263.78 mfu=26.67% step_ms=278.95 sm_util=81.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=60 loss=2079.1482 pred=2076.0654 aux=154.1343 lr=6.00e-06 toks/s=468,209 tflops=263.36 mfu=26.63% step_ms=279.34 sm_util=83.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=70 loss=2021.9357 pred=2018.7821 aux=157.6766 lr=7.00e-06 toks/s=470,851 tflops=264.84 mfu=26.78% step_ms=277.72 sm_util=82.0% hbm_util=59.0% mem_ctrl_util=59.0%
step=80 loss=1985.0073 pred=1981.7683 aux=161.9515 lr=8.00e-06 toks/s=479,761 tflops=269.86 mfu=27.29% step_ms=272.66 sm_util=93.0% hbm_util=68.0% mem_ctrl_util=68.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=51.74
```
## scatter_compile_tile_64_

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
step=10 loss=2058.7749 pred=2055.5637 aux=160.5614 lr=1.00e-06 toks/s=124,588 tflops=70.08 mfu=7.09% step_ms=904.32 sm_util=97.0% hbm_util=76.0% mem_ctrl_util=76.0%
step=20 loss=2101.1418 pred=2098.0312 aux=155.5314 lr=2.00e-06 toks/s=473,154 tflops=266.14 mfu=26.91% step_ms=276.52 sm_util=96.0% hbm_util=66.0% mem_ctrl_util=66.0%
step=30 loss=2098.6355 pred=2095.5483 aux=154.3569 lr=3.00e-06 toks/s=480,452 tflops=270.25 mfu=27.33% step_ms=272.16 sm_util=85.0% hbm_util=62.0% mem_ctrl_util=62.0%
step=40 loss=2025.1165 pred=2021.9033 aux=160.6590 lr=4.00e-06 toks/s=470,480 tflops=264.64 mfu=26.76% step_ms=277.96 sm_util=83.0% hbm_util=60.0% mem_ctrl_util=60.0%
step=50 loss=2085.2200 pred=2082.1206 aux=154.9633 lr=5.00e-06 toks/s=470,383 tflops=264.58 mfu=26.75% step_ms=278.10 sm_util=80.0% hbm_util=57.0% mem_ctrl_util=57.0%
step=60 loss=2082.3450 pred=2079.2224 aux=156.1323 lr=6.00e-06 toks/s=468,558 tflops=263.56 mfu=26.65% step_ms=279.13 sm_util=80.0% hbm_util=56.0% mem_ctrl_util=56.0%
step=70 loss=2033.8855 pred=2030.7004 aux=159.2525 lr=7.00e-06 toks/s=471,554 tflops=265.24 mfu=26.82% step_ms=277.29 sm_util=79.0% hbm_util=58.0% mem_ctrl_util=58.0%
step=80 loss=1998.9333 pred=1995.6567 aux=163.8299 lr=8.00e-06 toks/s=480,331 tflops=270.18 mfu=27.32% step_ms=272.34 sm_util=90.0% hbm_util=65.0% mem_ctrl_util=65.0%
run model=NVIDIA H100 80GB HBM3 precision=bf16 peak_vram_gb=51.74
```

