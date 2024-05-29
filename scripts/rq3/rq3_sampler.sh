#!/bin/bash
tuple_ls=("adult,income" "lung_cancer,dysp" "cylinder-bands,band_type" "diabetes,class" "contraceptive,y" "blood-transfusion-service-center,Class" "steel-plates-fault,target" "jungle_chess_2pcs_raw_endgame_complete,class" "telco_customer_churn,Churn" "bank-marketing,Class" "PhishingWebsites,Result" "hotel_bookings,booking_status")

for tuple in "${tuple_ls[@]}"
do
    IFS=',' read -r dataset_name label_name <<< "$tuple"
    python scripts/rq3/rq3_sampler.py -d "$dataset_name" -l "$label_name" --enable_auxiliary_sampler 1
    python scripts/rq3/rq3_sampler.py -d "$dataset_name" -l "$label_name" --enable_auxiliary_sampler 0
done


