SELECT version,
       COUNT(*)                          AS users,
       ROUND(AVG(retention_1::INT), 4)   AS ret_1day,
       ROUND(AVG(retention_7::INT), 4)   AS ret_7day,
       ROUND(AVG(sum_gamerounds), 2)     AS avg_rounds,
       MEDIAN(sum_gamerounds)            AS median_rounds
FROM ml_project.public.cookie_cats
WHERE sum_gamerounds < 10000
GROUP BY version;

WITH g AS (
  SELECT version, COUNT(*) AS n,
         AVG(retention_1::INT) AS p1, AVG(retention_7::INT) AS p7
  FROM ml_project.public.cookie_cats
  WHERE sum_gamerounds < 10000
  GROUP BY version
),
t AS (
  SELECT a.n AS n1, b.n AS n2, a.p1 AS a1, b.p1 AS b1, a.p7 AS a7, b.p7 AS b7
  FROM g a, g b
  WHERE a.version = 'gate_30' AND b.version = 'gate_40'
),
m AS (
  SELECT '7-day' AS metric, n1, n2, a7 AS pa, b7 AS pb FROM t
  UNION ALL
  SELECT '1-day', n1, n2, a1, b1 FROM t
)
SELECT metric,
       ROUND(pb - pa, 4) AS diff_gate40_minus_gate30,
       ROUND((pb - pa) - 1.96 * SQRT(pa*(1-pa)/n1 + pb*(1-pb)/n2), 4) AS ci_low,
       ROUND((pb - pa) + 1.96 * SQRT(pa*(1-pa)/n1 + pb*(1-pb)/n2), 4) AS ci_high,
       ROUND((pb - pa) / SQRT(((pa*n1 + pb*n2)/(n1+n2)) * (1 - (pa*n1 + pb*n2)/(n1+n2)) * (1/n1 + 1/n2)), 3) AS z_score
FROM m;