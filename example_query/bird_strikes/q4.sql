SELECT avg(CASE WHEN bird_strikes.REMAINS_COLLECTED == 'TRUE' THEN 1 ELSE 0 END) FROM bird_strikes.noisy GROUP BY M1
M1: bird_strikes-b52d7039-2da2-4e1c-a502-fc079bf64fe4, autogluon