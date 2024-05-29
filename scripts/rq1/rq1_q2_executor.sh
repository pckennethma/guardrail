#!/bin/bash

dataset_name_ls=("hotel_bookings" "adult" "lung_cancer" "cylinder-bands" "diabetes" "contraceptive" "blood-transfusion-service-center" "steel-plates-fault" "jungle_chess_2pcs_raw_endgame_complete" "telco_customer_churn" "bank-marketing" "PhishingWebsites")

# 循环遍历 dataset_name_ls 列表中的值
for dataset_name in "${dataset_name_ls[@]}"
do
    # 执行指令，替换 dataset_name
    python nsyn/app/q2_executor.py -p example_query/default/"$dataset_name".sql -n 05
done