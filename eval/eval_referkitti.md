### OG clip
Total expressions processed: 2

============================================================
📊 Evaluating RMOT-style Results (per expression)
============================================================

📂 Found 2 GT files and 2 result files

===== 0001_expr0000 =====
              num_frames   MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0000        403 -87.3% 0.177 23.9% 20.1% 29.4% 20.1% 29.4%   0  0  4  9   1 449 272         385         113

===== 0001_expr0001 =====
              num_frames   MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0001        419 -18.2% 0.176 29.5% 33.6% 26.2% 38.7% 30.2%   4  0  6 12   4 288 421         603         178

====== AVERAGE ======
               num_frames      mota     motp      idf1       idp  ...  num_fragmentations  num_false_positives  num_misses  num_objects  num_matches
0001_expr0000       403.0 -0.872727  0.17697  0.238648  0.201068  ...                 1.0                449.0       272.0        385.0        113.0
0001_expr0001       419.0 -0.182421  0.17587  0.294501  0.336170  ...                 4.0                288.0       421.0        603.0        178.0
AVG                 411.0 -0.527574  0.17642  0.266575  0.268619  ...                 2.5                368.5       346.5        494.0        145.5

[3 rows x 17 columns]

============================================================
📈 RMOT Summary over expressions: MOTA=-52.76% | IDF1=26.66%
============================================================

OPTUNA_RMOT:MOTA=-0.527574 IDF1=0.266575

✅ Complete! Results saved to: outputs/referkitti_rmot_2025-12-09_2250

# with text gate enabled default
Total expressions processed: 2

============================================================
📊 Evaluating RMOT-style Results (per expression)
============================================================

📂 Found 2 GT files and 2 result files

===== 0001_expr0000 =====
              num_frames   MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0000        410 -94.0% 0.175 24.0% 19.7% 30.6% 19.7% 30.6%   0  0  4  9   0 480 267         385         118

===== 0001_expr0001 =====
              num_frames   MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0001        419 -17.9% 0.177 30.3% 34.3% 27.2% 38.9% 30.8%   2  0  6 12   3 292 417         603         184

====== AVERAGE ======
               num_frames      mota      motp      idf1  ...  num_false_positives  num_misses  num_objects  num_matches
0001_expr0000       410.0 -0.940260  0.175091  0.240081  ...                480.0       267.0        385.0        118.0
0001_expr0001       419.0 -0.179104  0.177212  0.303423  ...                292.0       417.0        603.0        184.0
AVG                 414.5 -0.559682  0.176152  0.271752  ...                386.0       342.0        494.0        151.0

[3 rows x 17 columns]

============================================================
📈 RMOT Summary over expressions: MOTA=-55.97% | IDF1=27.18%
============================================================

OPTUNA_RMOT:MOTA=-0.559682 IDF1=0.271752

✅ Complete! Results saved to: outputs/referkitti_rmot_2025-12-11_2113

# text gate hard, text gate weight 0.7
Same results

# spatial filter with clip

Total expressions processed: 2

============================================================
📊 Evaluating RMOT-style Results (per expression)
============================================================

📂 Found 2 GT files and 2 result files

===== 0001_expr0000 =====
              num_frames   MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0000        369 -40.3% 0.175 30.4% 30.2% 30.6% 30.2% 30.6%   0  0  4  9   0 273 267         385         118

===== 0001_expr0001 =====
              num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0001        405 -5.1% 0.177 32.7% 40.9% 27.2% 46.4% 30.8%   2  0  6 12   3 215 417         603         184

====== AVERAGE ======
               num_frames      mota      motp      idf1  ...  num_false_positives  num_misses  num_objects  num_matches
0001_expr0000       369.0 -0.402597  0.175091  0.304124  ...                273.0       267.0        385.0        118.0
0001_expr0001       405.0 -0.051410  0.177212  0.326693  ...                215.0       417.0        603.0        184.0
AVG                 387.0 -0.227004  0.176152  0.315408  ...                244.0       342.0        494.0        151.0

[3 rows x 17 columns]

============================================================
📈 RMOT Summary over expressions: MOTA=-22.70% | IDF1=31.54%
============================================================

OPTUNA_RMOT:MOTA=-0.227004 IDF1=0.315408

# throw away clip, using gdino scores, referring thresh 0.40

Total expressions processed: 2

============================================================
📊 Evaluating RMOT-style Results (per expression)
============================================================

📂 Found 2 GT files and 2 result files

===== 0001_expr0000 =====
              num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0000        291 -8.6% 0.155 26.7% 41.1% 19.7% 41.1% 19.7%   0  0  6  7   0 109 309         385          76

===== 0001_expr0001 =====
              num_frames MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP  FN num_objects num_matches
0001_expr0001        390 5.6% 0.179 34.1% 54.2% 24.9% 56.3% 25.9%   1  0  6 12   1 121 447         603         155

====== AVERAGE ======
               num_frames      mota      motp      idf1  ...  num_false_positives  num_misses  num_objects  num_matches
0001_expr0000       291.0 -0.085714  0.155282  0.266667  ...                109.0       309.0        385.0         76.0
0001_expr0001       390.0  0.056385  0.178705  0.340909  ...                121.0       447.0        603.0        155.0
AVG                 340.5 -0.014665  0.166993  0.303788  ...                115.0       378.0        494.0        115.5

[3 rows x 17 columns]

============================================================
📈 RMOT Summary over expressions: MOTA=-1.47% | IDF1=30.38%
============================================================

OPTUNA_RMOT:MOTA=-0.014665 IDF1=0.303788

# exp 62-64 with color filtering

============================================================
📊 Evaluating RMOT-style Results (per expression)
============================================================

📂 Found 2 GT files and 2 result files

===== 0001_expr0060 =====
              num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM FP   FN num_objects num_matches
0001_expr0060        428 16.0% 0.167 31.3% 85.1% 19.1% 85.9% 19.3%   2  0 34 18   5 55 1395        1729         332

===== 0001_expr0061 =====
              num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM FP  FN num_objects num_matches
0001_expr0061        361 17.2% 0.172 35.0% 73.5% 23.0% 78.5% 24.6%   6  1 18  9  13 59 662         878         210

====== AVERAGE ======
               num_frames      mota      motp      idf1  ...  num_false_positives  num_misses  num_objects  num_matches
0001_expr0060       428.0  0.160208  0.166765  0.312559  ...                 55.0      1395.0       1729.0        332.0
0001_expr0061       361.0  0.171982  0.172326  0.350390  ...                 59.0       662.0        878.0        210.0
AVG                 394.5  0.166095  0.169546  0.331475  ...                 57.0      1028.5       1303.5        271.0

[3 rows x 17 columns]

============================================================
📈 RMOT Summary over expressions: MOTA=16.61% | IDF1=33.15%
============================================================

OPTUNA_RMOT:MOTA=0.166095 IDF1=0.331475

✅ Complete! Results saved to: outputs/referkitti_rmot_2025-12-12_0231

[I 2025-12-16 17:38:45,490] Trial 532 finished with value: 0.786026 and parameters: {'box_threshold': 0.4549706853844193, 'text_threshold': 0.36310447842781074, 'track_thresh': 0.15937989639421643, 'match_thresh': 0.8798305144582814, 'track_buffer': 110, 'lambda_weight': 0.5681541002030135, 'use_clip_in_high': False, 'use_clip_in_low': False, 'text_gate_mode': 'penalty', 'text_gate_weight': 0.735809202112974, 'referring_thresh': 0.26267060145602306, 'small_box_area_thresh': 4000, 'use_color_filter': True, 'use_spatial_filter': True}. Best is trial 518 with value: 0.7811466.
[Trial] Combined=0.1868 (MOTA=0.0015, IDF1=0.4649)
   Params: box=0.421, text=0.343, track=0.159, match=0.843, ref_thresh=0.244, clip_high=False, clip_low=False

   
