#!/bin/bash
tuple_ls=("adult,income" "lung_cancer,dysp" "cylinder-bands,band_type" "diabetes,class" "contraceptive,y" "blood-transfusion-service-center,Class" "steel-plates-fault,target" "jungle_chess_2pcs_raw_endgame_complete,class" "telco_customer_churn,Churn" "bank-marketing,Class" "PhishingWebsites,Result" "hotel_bookings,booking_status")

epsilon_ls=("0.01" "0.05" "0.1" "0.15" "0.2" "0.25")

# 循环遍历 tuple_ls 列表中的值
for tuple in "${tuple_ls[@]}"
do
    for epsilon in "${epsilon_ls[@]}"
    do
        # 使用逗号分隔符拆分 tuple 为 dataset_name 和 label_name
        IFS=',' read -r dataset_name label_name <<< "$tuple"
        
        # 执行指令，替换 dataset_name 和 label_name
        python scripts/rq3/rq3_epsilon.py -d "$dataset_name" -l "$label_name" -e "$epsilon"
    done
done
