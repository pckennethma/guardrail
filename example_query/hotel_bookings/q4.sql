SELECT avg(CASE WHEN M1 == 'Not_Canceled' THEN 1 ELSE 0 END) FROM hotel_bookings.noisy
M1: hotel_bookings-6fbdfb40-d3c6-4714-bb8f-3e2f842c387c, autogluon