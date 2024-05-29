#!/bin/bash
mode_ls=("gt" "rectify" "coerce" "ignore")
dataset_name_ls=("PhishingWebsites" "hotel_bookings" "adult" "lung_cancer" "cylinder-bands" "blood-transfusion-service-center" "contraceptive" "diabetes" "jungle_chess_2pcs_raw_endgame_complete" "steel-plates-fault" "telco_customer_churn" "bank-marketing,Class")

for dataset_name in "${dataset_name_ls[@]}"
do
    for mode in "${mode_ls[@]}"
    do
        if [ "$mode" != "gt" ]; then
            export NSYN_Q2_ERROR_HANDLING=$mode
        fi
        python scripts/rq2/rq2.py -p example_query/"$dataset_name" -m "$mode"
    done
done