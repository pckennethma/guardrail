SELECT avg(CASE WHEN telco_customer_churn.gender == 'Male' THEN 1 ELSE 0 END) FROM telco_customer_churn.noisy GROUP BY M1
M1: telco_customer_churn-17b6d6ca-ac32-4690-b58e-9bd494ebfa97, autogluon