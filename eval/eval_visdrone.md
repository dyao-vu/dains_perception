### python3 eval/eval_visdrone.py --data_root /isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone_mot_format --split val   
--box_threshold 0.4   --text_threshold 0.8   --track_thresh 0.45   --match_thresh 0.85    --text_prompt "car. pedestrian." --jobs 2 
--devices 0,1 --fp16 --tracker clip --weights /isis/home/hasana3/vlmtest/GroundingDINO/weights/swinb_aggro_visdrone_ft.pth --frame_rate 24 --use_clip_in_low --use_clip_in_unconf --lambda_weight 0.25 --text_sim_thresh 0.1 
###

📂 Found 7 GT files and 7 result files

===== uav0000086_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000086_00000_v        464 66.6% 0.249 77.1% 82.3% 72.5% 87.9% 77.5%  40 44 10 26  502 2401 5048       22410       17322

===== uav0000117_02622_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000117_02622_v        349 39.5% 0.238 50.8% 65.7% 41.4% 82.4% 51.8% 191 28 79 27  302 1690 7335       15224        7698

===== uav0000137_00458_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000137_00458_v        233 48.0% 0.224 62.2% 69.1% 56.5% 80.4% 65.8% 363 61 37 62  534 3385 7232       21118       13523

===== uav0000182_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000182_00000_v        363 14.1% 0.253 50.1% 52.1% 48.1% 57.8% 53.4%  33 28 30 28  132 3696 4425        9494        5036

===== uav0000268_05773_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP    FN num_objects num_matches
uav0000268_05773_v        978 22.1% 0.128 37.5% 95.0% 23.3% 95.0% 23.3%   0  5 38  7  13 160 10018       13068        3050

===== uav0000305_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000305_00000_v        184 48.9% 0.154 71.1% 74.7% 67.9% 77.7% 70.6%  69 26 12 13  64 982 1421        4839        3349

===== uav0000339_00001_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000339_00001_v        275 42.4% 0.232 53.3% 70.0% 43.0% 85.1% 52.2%  74 29 20 16  156 930 4872       10203        5257

====== AVERAGE ======
                    num_frames      mota      motp      idf1       idp  ...  num_fragmentations  num_false_positives    num_misses   num_objects   num_matches
uav0000086_00000_v  464.000000  0.665819  0.248812  0.770967  0.822598  ...          502.000000               2401.0   5048.000000  22410.000000  17322.000000
uav0000117_02622_v  349.000000  0.394640  0.237996  0.507842  0.657480  ...          302.000000               1690.0   7335.000000  15224.000000   7698.000000
uav0000137_00458_v  233.000000  0.480064  0.223879  0.621949  0.691216  ...          534.000000               3385.0   7232.000000  21118.000000  13523.000000
uav0000182_00000_v  363.000000  0.141142  0.252782  0.500575  0.521392  ...          132.000000               3696.0   4425.000000   9494.000000   5036.000000
uav0000268_05773_v  978.000000  0.221151  0.128237  0.374739  0.950156  ...           13.000000                160.0  10018.000000  13068.000000   3050.000000
uav0000305_00000_v  184.000000  0.489151  0.154090  0.711116  0.746591  ...           64.000000                982.0   1421.000000   4839.000000   3349.000000
uav0000339_00001_v  275.000000  0.424091  0.232065  0.532677  0.700367  ...          156.000000                930.0   4872.000000  10203.000000   5257.000000
AVG                 406.571429  0.402294  0.211123  0.574266  0.727114  ...          243.285714               1892.0   5764.428571  13765.142857   7890.714286

[8 rows x 17 columns]

============================================================
📈 Summary: MOTA=40.23% | IDF1=57.43%
============================================================

OPTUNA:MOTA=0.402294 IDF1=0.574266

✅ Complete! Results saved to: outputs/visdrone_val_2025-11-11_0641

# with byettrack same params
📂 Found 7 GT files and 7 result files

===== uav0000086_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000086_00000_v        464 66.6% 0.249 77.1% 82.3% 72.5% 87.9% 77.4%  39 44 10 26  502 2393 5060       22410       17311

===== uav0000117_02622_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000117_02622_v        349 39.1% 0.237 51.4% 67.0% 41.7% 82.4% 51.3% 171 27 79 28  282 1672 7421       15224        7632

===== uav0000137_00458_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000137_00458_v        233 48.2% 0.223 62.1% 69.7% 56.0% 80.8% 65.0% 289 57 39 64  497 3263 7395       21118       13434

===== uav0000182_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000182_00000_v        363 16.0% 0.251 50.6% 53.7% 47.7% 59.2% 52.6%  31 28 33 25  124 3445 4500        9494        4963

===== uav0000268_05773_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP    FN num_objects num_matches
uav0000268_05773_v        978 22.1% 0.128 37.5% 95.0% 23.3% 95.0% 23.3%   0  5 38  7  13 160 10018       13068        3050

===== uav0000305_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000305_00000_v        184 48.4% 0.148 72.0% 77.0% 67.6% 77.8% 68.3%  18 23 16 12  35 945 1533        4839        3288

===== uav0000339_00001_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000339_00001_v        275 42.5% 0.231 53.5% 71.4% 42.7% 86.0% 51.4%  56 27 22 16  148 853 4955       10203        5192

====== AVERAGE ======
                    num_frames      mota      motp      idf1       idp  ...  num_fragmentations  num_false_positives    num_misses   num_objects   num_matches
uav0000086_00000_v  464.000000  0.665685  0.248821  0.770811  0.822874  ...          502.000000          2393.000000   5060.000000  22410.000000  17311.000000
uav0000117_02622_v  349.000000  0.391487  0.236624  0.513948  0.669868  ...          282.000000          1672.000000   7421.000000  15224.000000   7632.000000
uav0000137_00458_v  233.000000  0.481627  0.222628  0.621195  0.696750  ...          497.000000          3263.000000   7395.000000  21118.000000  13434.000000
uav0000182_00000_v  363.000000  0.159890  0.250671  0.505548  0.537149  ...          124.000000          3445.000000   4500.000000   9494.000000   4963.000000
uav0000268_05773_v  978.000000  0.221151  0.128237  0.374739  0.950156  ...           13.000000           160.000000  10018.000000  13068.000000   3050.000000
uav0000305_00000_v  184.000000  0.484191  0.147539  0.719912  0.769701  ...           35.000000           945.000000   1533.000000   4839.000000   3288.000000
uav0000339_00001_v  275.000000  0.425267  0.231077  0.534715  0.714473  ...          148.000000           853.000000   4955.000000  10203.000000   5192.000000
AVG                 406.571429  0.404186  0.209371  0.577267  0.737282  ...          228.714286          1818.714286   5840.285714  13765.142857   7838.571429

[8 rows x 17 columns]

============================================================
📈 Summary: MOTA=40.42% | IDF1=57.73%
============================================================

OPTUNA:MOTA=0.404186 IDF1=0.577267

✅ Complete! Results saved to: outputs/visdrone_val_2025-11-11_0653
