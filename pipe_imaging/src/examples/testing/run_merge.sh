#!/bin/bash

directory_in=$1 #directory input where the prediction files are
option=$2 #data retrieval mode when running simulation
run_mode=$3 #data retrieval mode when training datasets
epoch=$4 #iteration number
directory_out=$5 #directory output to store the merged files
directory_log=$6 #directory to store the log file

./merge_predict_file.py $directory_in $option $run_mode 14 15 $epoch $directory_out/corr_file_conc_1_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 43 45 $epoch $directory_out/corr_file_conc_2_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 11 12 $epoch $directory_out/corr_file_conc_3_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 46 48 $epoch $directory_out/corr_file_conc_4_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 49 51 $epoch $directory_out/corr_file_conc_5_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 7 9 $epoch $directory_out/corr_file_conc_6_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 53 54 $epoch $directory_out/corr_file_conc_7_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode	2 3 $epoch $directory_out/corr_file_conc_8_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 55 57 $epoch $directory_out/corr_file_conc_9_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 5 6 $epoch $directory_out/corr_file_conc_10_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 58 60 $epoch $directory_out/corr_file_conc_11_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 61 63 $epoch $directory_out/corr_file_conc_12_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 64 66 $epoch $directory_out/corr_file_conc_13_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 67 69 $epoch $directory_out/corr_file_conc_14_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 71 72 $epoch $directory_out/corr_file_conc_15_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 73 75 $epoch $directory_out/corr_file_conc_16_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 76 78 $epoch $directory_out/corr_file_conc_17_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 80 81 $epoch $directory_out/corr_file_conc_18_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 101 102 $epoch $directory_out/corr_file_conc_0pt5_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 103 105 $epoch $directory_out/corr_file_conc_0pt4_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 107 108 $epoch $directory_out/corr_file_conc_0pt3_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 109 111 $epoch $directory_out/corr_file_conc_0pt2_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 113 114 $epoch $directory_out/corr_file_conc_0pt1_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
./merge_predict_file.py $directory_in $option $run_mode 116 117 $epoch $directory_out/corr_file_conc_0pt01_${option}_${run_mode}_${epoch}_predict.csv $directory_log/merge_predict_concentric_log.txt
