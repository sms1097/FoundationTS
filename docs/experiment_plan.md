# Experirements

Goal is to measure perf of the baseline architecture, then see how much we can improve from the baseline.

Metrics will be:

- Compare at 25, 50, and 100% of compute budget (GPU Hours)
- Compare time to cross threshold of metrics (MSE, MAE)

Things to test:

- **baseline:** TimeMOE with pointwise tokenization
- Patching
- Sliding window attention
- QK normalization
- FP8 matmuls
- Batch size schedule
- Gradient Compression (SquishGrad)

Model size: 50M active?

## Setup

- data mixture + sampling:
  - TimeMOE dataset
  - Fev Evaluation datasets
- context length: 4096 (for variants run at 512)
- horizon distribution: [1, 8, 32, 64]
- train budget: 2 GPU hours (H100)
- metrics: overall + horizon-binned + per-frequency bucket
- 3 fixed seeds for final comparisons (use 1 seed for early triage)

Always log:

- cumulative GPU-seconds
- tokens/time-points processed
- effective sequence length (patching changes this)
- train stability stats (grad norm, loss spikes, expert tokens and router collapses)
- throughput (tokens/sec) + peak memory

## Architecture and Modeling

- Primary model changes to experiment with are the following:
  - patching
  - attention variants
  - QK normalization
- These will run on 10% training budget.
- Then follow up with pairwise interactions for the modeling.
- Scale up to 4096 after we figure out which variable works best
- Throw in some FP8 training
- Batch size schedule

## Reporting

- metric vs GPU-hours curve (one plot)
- metric at fixed compute points (0.25×, 0.5×, 1×)
  - Compare early changes at 10% compute budget
- compute-to-target table (time/GPU-hours to reach baseline quality)
- throughput + memory + stability notes


## Baseline


```
foundationts train \
  --dataset-path time300b_selected \
  --microbatch-size 32 \
  --global-batch-size 1024 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --num-expert-layers 1 \
  --hidden-size 384 \
  --n-head 12 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 1536 \
  --d-expert 768 \
  --log-every 10 \
  --checkpoint-every 0 \
  --mfu-peak-tflops 989 \
  --max-wall-time-s 720 \
  --final-val-on-budget \
  --val-every 1000 \
  --run-name baseline
```

```
params total=113.35M (113,351,913) active=49.65M (49,650,921)
device model=NVIDIA H100 80GB HBM3 precision=bf16
budget seconds=720.0
step=10 loss=3.9271 pred=0.5979 aux=166.4627 mae=0.9077 mse=1.3259 lr=1.00e-06 toks/s=288,049 mfu=15.27%
val step=10 pred=0.6044 aux=151.7858 mae=0.9091 mse=1.3585
step=20 loss=3.7370 pred=0.5556 aux=159.0689 mae=0.8866 mse=1.2895 lr=2.00e-06 toks/s=337,917 mfu=17.91%
val step=20 pred=0.5622 aux=147.5710 mae=0.8934 mse=1.3143
step=30 loss=3.5837 pred=0.5085 aux=153.7627 mae=0.8670 mse=1.2243 lr=3.00e-06 toks/s=340,097 mfu=18.03%
val step=30 pred=0.5089 aux=143.8041 mae=0.8665 mse=1.2437
step=40 loss=3.3511 pred=0.4480 aux=145.1548 mae=0.8340 mse=1.1377 lr=4.00e-06 toks/s=360,782 mfu=19.12%
val step=40 pred=0.4534 aux=143.6650 mae=0.8367 mse=1.1692
step=50 loss=3.3055 pred=0.3950 aux=145.5279 mae=0.8020 mse=1.0601 lr=5.00e-06 toks/s=336,276 mfu=17.83%
val step=50 pred=0.3974 aux=144.8716 mae=0.8003 mse=1.0837
budget hit: stopping before step=55
val step=54 pred=0.3815 aux=143.0658 mae=0.7878 mse=1.0564
```

## Patching (high resolution)
```
foundationts train \
  --dataset-path time300b_selected \
  --microbatch-size 896 \
  --global-batch-size 1792 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --patch \
  --patch-len 32 \
  --patch-stride 16 \
  --num-expert-layers 1 \
  --hidden-size 384 \
  --n-head 12 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 1536 \
  --d-expert 768 \
  --log-every 10 \
  --checkpoint-every 0 \
  --mfu-peak-tflops 989 \
  --max-wall-time-s 720 \
  --final-val-on-budget \
  --val-every 10 \
  --run-name patch_32_16

params total=113.38M (113,375,721) active=49.67M (49,674,729)
device model=NVIDIA H100 80GB HBM3 precision=bf16
budget seconds=720.0
step=10 loss=3.6560 pred=0.5776 aux=153.9194 mae=0.8857 mse=1.2762 lr=1.00e-06 toks/s=4,240,381 mfu=8.33%
val step=10 pred=0.5755 aux=147.9991 mae=0.8863 mse=1.2754
step=20 loss=3.5328 pred=0.5512 aux=149.0757 mae=0.8786 mse=1.2613 lr=2.00e-06 toks/s=4,267,306 mfu=8.38%
val step=20 pred=0.5486 aux=145.9348 mae=0.8790 mse=1.2553
step=30 loss=3.4081 pred=0.5182 aux=144.4931 mae=0.8674 mse=1.2225 lr=3.00e-06 toks/s=4,344,465 mfu=8.53%
val step=30 pred=0.5147 aux=144.2087 mae=0.8662 mse=1.2214
step=40 loss=3.3688 pred=0.4834 aux=144.2687 mae=0.8498 mse=1.1748 lr=4.00e-06 toks/s=4,467,127 mfu=8.77%
val step=40 pred=0.4803 aux=144.0637 mae=0.8469 mse=1.1714
step=50 loss=3.3291 pred=0.4477 aux=144.0714 mae=0.8249 mse=1.1168 lr=5.00e-06 toks/s=4,546,890 mfu=8.93%
val step=50 pred=0.4489 aux=143.9516 mae=0.8265 mse=1.1198
step=60 loss=3.3034 pred=0.4227 aux=144.0356 mae=0.8076 mse=1.0703 lr=6.00e-06 toks/s=4,462,266 mfu=8.76%
val step=60 pred=0.4239 aux=144.0121 mae=0.8090 mse=1.0771
step=70 loss=3.2841 pred=0.4039 aux=144.0096 mae=0.7950 mse=1.0372 lr=7.00e-06 toks/s=4,769,679 mfu=9.37%
val step=70 pred=0.4027 aux=144.0009 mae=0.7935 mse=1.0406
step=80 loss=3.2719 pred=0.3917 aux=144.0083 mae=0.7869 mse=1.0258 lr=8.00e-06 toks/s=4,647,254 mfu=9.13%
val step=80 pred=0.3844 aux=143.9903 mae=0.7786 mse=1.0085
step=90 loss=3.2507 pred=0.3706 aux=144.0028 mae=0.7660 mse=0.9816 lr=9.00e-06 toks/s=4,856,178 mfu=9.54%
val step=90 pred=0.3686 aux=143.9856 mae=0.7645 mse=0.9804
step=100 loss=3.2494 pred=0.3693 aux=144.0046 mae=0.7669 mse=0.9892 lr=1.00e-05 toks/s=4,882,843 mfu=9.59%
val step=100 pred=0.3551 aux=143.9872 mae=0.7507 mse=0.9537
step=110 loss=3.2200 pred=0.3399 aux=144.0085 mae=0.7324 mse=0.9196 lr=1.10e-05 toks/s=4,789,674 mfu=9.41%
val step=110 pred=0.3436 aux=143.9893 mae=0.7371 mse=0.9285
step=120 loss=3.2099 pred=0.3297 aux=144.0078 mae=0.7188 mse=0.8992 lr=1.20e-05 toks/s=4,872,408 mfu=9.57%
val step=120 pred=0.3336 aux=143.9878 mae=0.7247 mse=0.9053
step=130 loss=3.2053 pred=0.3252 aux=144.0033 mae=0.7156 mse=0.8859 lr=1.30e-05 toks/s=4,876,093 mfu=9.58%
val step=130 pred=0.3254 aux=144.0070 mae=0.7133 mse=0.8863
step=140 loss=3.1974 pred=0.3174 aux=144.0030 mae=0.7043 mse=0.8677 lr=1.40e-05 toks/s=5,224,587 mfu=10.26%
val step=140 pred=0.3187 aux=143.9858 mae=0.7050 mse=0.8712
step=150 loss=3.1918 pred=0.3115 aux=144.0129 mae=0.6951 mse=0.8557 lr=1.50e-05 toks/s=5,428,518 mfu=10.66%
val step=150 pred=0.3135 aux=143.9836 mae=0.6981 mse=0.8596
step=160 loss=3.1864 pred=0.3059 aux=144.0246 mae=0.6884 mse=0.8412 lr=1.60e-05 toks/s=5,695,701 mfu=11.19%
val step=160 pred=0.3088 aux=144.0178 mae=0.6919 mse=0.8500
step=170 loss=3.1853 pred=0.3050 aux=144.0132 mae=0.6856 mse=0.8420 lr=1.70e-05 toks/s=5,749,733 mfu=11.29%
val step=170 pred=0.3046 aux=143.9762 mae=0.6867 mse=0.8408
step=180 loss=3.1802 pred=0.3001 aux=144.0044 mae=0.6844 mse=0.8418 lr=1.80e-05 toks/s=5,698,325 mfu=11.19%
val step=180 pred=0.3008 aux=143.9808 mae=0.6819 mse=0.8320
step=190 loss=3.1790 pred=0.2988 aux=144.0099 mae=0.6736 mse=0.8386 lr=1.90e-05 toks/s=5,911,381 mfu=11.61%
val step=190 pred=0.2972 aux=143.9780 mae=0.6760 mse=0.8234
step=200 loss=3.1697 pred=0.2896 aux=144.0059 mae=0.6654 mse=0.8040 lr=2.00e-05 toks/s=5,981,178 mfu=11.75%
val step=200 pred=0.2945 aux=143.9996 mae=0.6713 mse=0.8141
step=210 loss=3.1733 pred=0.2931 aux=144.0093 mae=0.6694 mse=0.8095 lr=2.10e-05 toks/s=6,436,701 mfu=12.64%
val step=210 pred=0.2912 aux=144.0031 mae=0.6657 mse=0.8065
step=220 loss=3.1720 pred=0.2918 aux=144.0084 mae=0.6645 mse=0.8226 lr=2.20e-05 toks/s=6,573,379 mfu=12.91%
val step=220 pred=0.2884 aux=144.0303 mae=0.6609 mse=0.7996
step=230 loss=3.1667 pred=0.2866 aux=144.0068 mae=0.6560 mse=0.7963 lr=2.30e-05 toks/s=6,044,701 mfu=11.87%
val step=230 pred=0.2859 aux=144.0349 mae=0.6572 mse=0.7926
step=240 loss=3.1614 pred=0.2811 aux=144.0164 mae=0.6482 mse=0.7804 lr=2.40e-05 toks/s=6,240,888 mfu=12.26%
val step=240 pred=0.2842 aux=144.0034 mae=0.6525 mse=0.7869
step=250 loss=3.1613 pred=0.2810 aux=144.0150 mae=0.6441 mse=0.7854 lr=2.50e-05 toks/s=6,444,958 mfu=12.66%
val step=250 pred=0.2824 aux=144.0222 mae=0.6508 mse=0.7821
step=260 loss=3.1573 pred=0.2772 aux=144.0042 mae=0.6409 mse=0.7716 lr=2.60e-05 toks/s=6,576,974 mfu=12.92%
val step=260 pred=0.2806 aux=143.9721 mae=0.6480 mse=0.7778
step=270 loss=3.1628 pred=0.2822 aux=144.0278 mae=0.6509 mse=0.7873 lr=2.70e-05 toks/s=6,632,757 mfu=13.03%
val step=270 pred=0.2793 aux=143.9646 mae=0.6444 mse=0.7740
step=280 loss=3.1564 pred=0.2762 aux=144.0067 mae=0.6356 mse=0.7643 lr=2.80e-05 toks/s=6,575,990 mfu=12.92%
val step=280 pred=0.2791 aux=143.9934 mae=0.6422 mse=0.7710
step=290 loss=3.1626 pred=0.2820 aux=144.0295 mae=0.6454 mse=0.7861 lr=2.90e-05 toks/s=6,590,845 mfu=12.94%
val step=290 pred=0.2773 aux=144.0469 mae=0.6414 mse=0.7673
step=300 loss=3.1532 pred=0.2730 aux=144.0117 mae=0.6359 mse=0.7795 lr=3.00e-05 toks/s=6,710,189 mfu=13.18%
val step=300 pred=0.2760 aux=144.0387 mae=0.6377 mse=0.7648
step=310 loss=3.1519 pred=0.2716 aux=144.0128 mae=0.6349 mse=0.7591 lr=3.10e-05 toks/s=6,856,447 mfu=13.47%
val step=310 pred=0.2749 aux=144.0233 mae=0.6365 mse=0.7617
step=320 loss=3.1574 pred=0.2771 aux=144.0159 mae=0.6386 mse=0.7685 lr=3.20e-05 toks/s=6,579,720 mfu=12.92%
val step=320 pred=0.2734 aux=144.0369 mae=0.6352 mse=0.7583
step=330 loss=3.1494 pred=0.2691 aux=144.0119 mae=0.6230 mse=0.7716 lr=3.30e-05 toks/s=6,988,005 mfu=13.72%
val step=330 pred=0.2725 aux=144.0197 mae=0.6324 mse=0.7558
step=340 loss=3.1507 pred=0.2706 aux=144.0043 mae=0.6296 mse=0.7520 lr=3.40e-05 toks/s=6,590,006 mfu=12.94%
val step=340 pred=0.2718 aux=144.0105 mae=0.6311 mse=0.7532
step=350 loss=3.1549 pred=0.2749 aux=144.0004 mae=0.6292 mse=0.7714 lr=3.50e-05 toks/s=6,188,128 mfu=12.15%
val step=350 pred=0.2709 aux=143.9749 mae=0.6289 mse=0.7516
step=360 loss=3.1493 pred=0.2691 aux=144.0101 mae=0.6242 mse=0.7444 lr=3.60e-05 toks/s=6,657,112 mfu=13.07%
val step=360 pred=0.2708 aux=144.0584 mae=0.6335 mse=0.7499
step=370 loss=3.1524 pred=0.2722 aux=144.0124 mae=0.6316 mse=0.7623 lr=3.70e-05 toks/s=7,118,265 mfu=13.98%
val step=370 pred=0.2696 aux=144.0464 mae=0.6277 mse=0.7464
step=380 loss=3.1566 pred=0.2762 aux=144.0231 mae=0.6331 mse=0.7654 lr=3.80e-05 toks/s=7,316,203 mfu=14.37%
val step=380 pred=0.2689 aux=144.0488 mae=0.6291 mse=0.7444
step=390 loss=3.1524 pred=0.2723 aux=144.0081 mae=0.6315 mse=0.7551 lr=3.90e-05 toks/s=7,028,157 mfu=13.80%
val step=390 pred=0.2678 aux=144.0072 mae=0.6240 mse=0.7426
step=400 loss=3.1455 pred=0.2651 aux=144.0205 mae=0.6208 mse=0.7398 lr=4.00e-05 toks/s=6,800,318 mfu=13.36%
val step=400 pred=0.2671 aux=144.0034 mae=0.6271 mse=0.7404
step=410 loss=3.1447 pred=0.2645 aux=144.0063 mae=0.6217 mse=0.7335 lr=4.10e-05 toks/s=7,121,888 mfu=13.99%
val step=410 pred=0.2671 aux=143.9998 mae=0.6235 mse=0.7381
step=420 loss=3.1440 pred=0.2637 aux=144.0115 mae=0.6207 mse=0.7449 lr=4.20e-05 toks/s=6,903,894 mfu=13.56%
val step=420 pred=0.2659 aux=144.0138 mae=0.6203 mse=0.7364
step=430 loss=3.1487 pred=0.2685 aux=144.0121 mae=0.6253 mse=0.7421 lr=4.30e-05 toks/s=7,192,987 mfu=14.13%
val step=430 pred=0.2651 aux=144.0042 mae=0.6218 mse=0.7344
step=440 loss=3.1392 pred=0.2591 aux=144.0095 mae=0.6121 mse=0.7183 lr=4.40e-05 toks/s=7,294,612 mfu=14.33%
val step=440 pred=0.2650 aux=144.0023 mae=0.6192 mse=0.7329
budget hit: stopping before step=441
val step=440 pred=0.2650 aux=144.0023 mae=0.6192 mse=0.7329
```


## Patching (Agressive)
```
foundationts train \
  --dataset-path time300b_selected \
  --microbatch-size 2048 \
  --global-batch-size 8192 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --patch \
  --patch-len 64 \
  --patch-stride 64 \
  --num-expert-layers 1 \
  --hidden-size 384 \
  --n-head 12 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 1536 \
  --d-expert 768 \
  --log-every 10 \
  --checkpoint-every 0 \
  --mfu-peak-tflops 989 \
  --max-wall-time-s 720 \
  --final-val-on-budget \
  --val-every 10 \
  --run-name patch_64_64
params total=113.40M (113,400,297) active=49.70M (49,699,305)
device model=NVIDIA H100 80GB HBM3 precision=bf16
budget seconds=720.0
step=10 loss=3.6765 pred=0.5755 aux=155.0475 mae=0.8907 mse=1.2932 lr=1.00e-06 toks/s=10,313,610 mfu=4.91%
val step=10 pred=0.5733 aux=147.9714 mae=0.8896 mse=1.2913
step=20 loss=3.5817 pred=0.5582 aux=151.1721 mae=0.8853 mse=1.2844 lr=2.00e-06 toks/s=11,026,083 mfu=5.25%
val step=20 pred=0.5517 aux=146.3465 mae=0.8805 mse=1.2665
step=30 loss=3.4544 pred=0.5310 aux=146.1703 mae=0.8674 mse=1.2415 lr=3.00e-06 toks/s=11,387,625 mfu=5.43%
val step=30 pred=0.5267 aux=144.6091 mae=0.8658 mse=1.2273
step=40 loss=3.3866 pred=0.5023 aux=144.2154 mae=0.8505 mse=1.1941 lr=4.00e-06 toks/s=12,321,461 mfu=5.87%
val step=40 pred=0.4979 aux=144.0276 mae=0.8474 mse=1.1787
step=50 loss=3.3562 pred=0.4728 aux=144.1688 mae=0.8292 mse=1.1436 lr=5.00e-06 toks/s=12,895,113 mfu=6.14%
val step=50 pred=0.4696 aux=144.1302 mae=0.8278 mse=1.1279
step=60 loss=3.3290 pred=0.4480 aux=144.0505 mae=0.8106 mse=1.0842 lr=6.00e-06 toks/s=13,270,940 mfu=6.32%
val step=60 pred=0.4453 aux=144.0259 mae=0.8096 mse=1.0828
step=70 loss=3.3054 pred=0.4246 aux=144.0423 mae=0.7930 mse=1.0431 lr=7.00e-06 toks/s=13,809,007 mfu=6.58%
val step=70 pred=0.4244 aux=143.9995 mae=0.7929 mse=1.0433
step=80 loss=3.2881 pred=0.4080 aux=144.0035 mae=0.7786 mse=1.0145 lr=8.00e-06 toks/s=14,153,945 mfu=6.74%
val step=80 pred=0.4061 aux=144.0116 mae=0.7781 mse=1.0080
step=90 loss=3.2790 pred=0.3990 aux=144.0041 mae=0.7721 mse=1.0098 lr=9.00e-06 toks/s=14,627,335 mfu=6.97%
val step=90 pred=0.3903 aux=143.9994 mae=0.7648 mse=0.9765
step=100 loss=3.2607 pred=0.3806 aux=144.0059 mae=0.7553 mse=0.9627 lr=1.00e-05 toks/s=14,051,996 mfu=6.70%
val step=100 pred=0.3764 aux=144.0067 mae=0.7516 mse=0.9484
step=110 loss=3.2434 pred=0.3634 aux=144.0031 mae=0.7379 mse=0.9278 lr=1.10e-05 toks/s=15,561,472 mfu=7.41%
val step=110 pred=0.3643 aux=144.0112 mae=0.7395 mse=0.9236
step=120 loss=3.2399 pred=0.3599 aux=144.0022 mae=0.7329 mse=0.9224 lr=1.20e-05 toks/s=15,560,591 mfu=7.41%
val step=120 pred=0.3538 aux=144.0060 mae=0.7287 mse=0.9020
step=130 loss=3.2268 pred=0.3466 aux=144.0100 mae=0.7202 mse=0.8882 lr=1.30e-05 toks/s=14,899,444 mfu=7.10%
val step=130 pred=0.3450 aux=144.0024 mae=0.7189 mse=0.8839
step=140 loss=3.2251 pred=0.3448 aux=144.0123 mae=0.7178 mse=0.8904 lr=1.40e-05 toks/s=16,783,460 mfu=8.00%
val step=140 pred=0.3381 aux=143.9992 mae=0.7113 mse=0.8703
step=150 loss=3.2145 pred=0.3344 aux=144.0045 mae=0.7070 mse=0.8672 lr=1.50e-05 toks/s=16,821,471 mfu=8.02%
val step=150 pred=0.3324 aux=143.9995 mae=0.7050 mse=0.8593
step=160 loss=3.2121 pred=0.3321 aux=144.0048 mae=0.7050 mse=0.8817 lr=1.60e-05 toks/s=17,382,375 mfu=8.28%
val step=160 pred=0.3271 aux=143.9988 mae=0.6996 mse=0.8494
step=170 loss=3.2011 pred=0.3210 aux=144.0065 mae=0.6907 mse=0.8406 lr=1.70e-05 toks/s=18,216,228 mfu=8.68%
val step=170 pred=0.3225 aux=144.0022 mae=0.6938 mse=0.8409
step=180 loss=3.2042 pred=0.3241 aux=144.0053 mae=0.6934 mse=0.8502 lr=1.80e-05 toks/s=18,459,977 mfu=8.80%
val step=180 pred=0.3183 aux=144.0077 mae=0.6889 mse=0.8326
step=190 loss=3.1969 pred=0.3167 aux=144.0089 mae=0.6846 mse=0.8301 lr=1.90e-05 toks/s=17,714,632 mfu=8.44%
val step=190 pred=0.3150 aux=143.9976 mae=0.6838 mse=0.8253
step=200 loss=3.1942 pred=0.3142 aux=144.0040 mae=0.6836 mse=0.8328 lr=2.00e-05 toks/s=18,820,344 mfu=8.97%
val step=200 pred=0.3119 aux=144.0147 mae=0.6798 mse=0.8182
step=210 loss=3.1884 pred=0.3083 aux=144.0059 mae=0.6766 mse=0.8157 lr=2.10e-05 toks/s=18,823,781 mfu=8.97%
val step=210 pred=0.3089 aux=143.9993 mae=0.6752 mse=0.8121
step=220 loss=3.1914 pred=0.3111 aux=144.0155 mae=0.6771 mse=0.8236 lr=2.20e-05 toks/s=19,482,695 mfu=9.28%
val step=220 pred=0.3066 aux=144.0007 mae=0.6727 mse=0.8065
step=230 loss=3.1859 pred=0.3058 aux=144.0028 mae=0.6700 mse=0.8140 lr=2.30e-05 toks/s=19,086,261 mfu=9.09%
val step=230 pred=0.3044 aux=144.0103 mae=0.6690 mse=0.8014
step=240 loss=3.1832 pred=0.3031 aux=144.0054 mae=0.6663 mse=0.8014 lr=2.40e-05 toks/s=19,657,671 mfu=9.37%
val step=240 pred=0.3022 aux=144.0182 mae=0.6657 mse=0.7968
step=250 loss=3.1854 pred=0.3053 aux=144.0070 mae=0.6692 mse=0.8123 lr=2.50e-05 toks/s=19,846,148 mfu=9.46%
val step=250 pred=0.3000 aux=143.9893 mae=0.6627 mse=0.7927
step=260 loss=3.1802 pred=0.3000 aux=144.0069 mae=0.6622 mse=0.7980 lr=2.60e-05 toks/s=19,621,488 mfu=9.35%
val step=260 pred=0.2982 aux=144.0217 mae=0.6602 mse=0.7890
step=270 loss=3.1771 pred=0.2967 aux=144.0209 mae=0.6571 mse=0.7866 lr=2.70e-05 toks/s=19,722,419 mfu=9.40%
val step=270 pred=0.2974 aux=143.9790 mae=0.6579 mse=0.7859
budget hit: stopping before step=276
val step=275 pred=0.2964 aux=143.9922 mae=0.6568 mse=0.7841
```
