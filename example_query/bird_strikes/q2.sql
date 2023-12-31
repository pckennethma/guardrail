SELECT avg(CASE WHEN M1 == 'S' THEN 1 ELSE 0 END) FROM bird_strikes.noisy WHERE bird_strikes.Time_of_day == 'Day'
M1: bird_strikes-b52d7039-2da2-4e1c-a502-fc079bf64fe4, autogluon
