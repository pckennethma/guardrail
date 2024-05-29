#!/bin/bash
tuple_ls=("adult,income" "lung_cancer,dysp" "contraceptive,y" "hotel_bookings,booking_status" "diabetes,class" "bank-marketing,Class" "blood-transfusion-service-center,Class" "PhishingWebsites,Result" "cylinder-bands,band_type" "steel-plates-fault,target" "jungle_chess_2pcs_raw_endgame_complete,class" "telco_customer_churn,Churn") 

# enumerate tuple_ls
for tuple in "${tuple_ls[@]}"
do
    IFS=',' read -r dataset_name label_name <<< "$tuple"
    python nsyn/app/ml_backend/autogluon_trainer.py -d "$dataset_name" -l "$label_name"
done