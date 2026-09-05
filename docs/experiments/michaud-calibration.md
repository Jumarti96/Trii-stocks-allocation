# michaud_spread calibration

Walk-forward sweep of `michaud_spread` on two arms. Arm A is a control on
the pre-change configuration (Huber loss, `periods_to_forecast=4`) and must
recover the deployed value of 4.0. If it does not, the harness is not
measuring the right thing and arm B's number cannot be trusted.

## Selection rule

Realised vol and turnover both fall **monotonically** in `s` (larger `s`
flattens weights toward equal-weight over the top-N), so neither can pick
an interior optimum on its own -- and Sharpe on gross returns degenerates
to `s=0`. The recommendation therefore maximises Sharpe after charging
turnover at a 0.5% round-trip cost, which is both
well-posed and the quantity that determines what the strategy earns.

Rebalance cadence 24 periods | n_runs 50 | mc_draws 1000 | spreads [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

## Control gate: VALIDATED

Arm A optimum at `s=2.0` (deployed: 4.0).

**Recommended `michaud_spread` for rank_ic / pto=24: `1.0`**

Bootstrap over the 13 rebalance periods: that value wins 42% of resamples.  **Below 50%, so the sample cannot separate these settings -- treat the value as a direction, not a precise recommendation.**

```
                  p_best  sharpe_se
arm       spread                   
A_control 0.0     0.2085   0.513838
          1.0     0.0995   0.516404
          2.0     0.3915   0.578203
          4.0     0.1105   0.667326
          6.0     0.1210   0.739052
          8.0     0.0005   0.721915
          12.0    0.0355   0.750772
          16.0    0.0330   0.701209
B_target  0.0     0.2500   0.467427
          1.0     0.4245   0.491672
          2.0     0.1225   0.474627
          4.0     0.1015   0.522111
          6.0     0.0025   0.539831
          8.0     0.0020   0.564825
          12.0    0.0425   0.620743
          16.0    0.0545   0.670952
```

## Full sweep

```
      arm  spread  ann_return  ann_vol   sharpe  turnover  turnover_drift  max_weight  effective_n    n_held  cost_drag  net_return  sharpe_net  worst_period  max_drawdown  downside_dev  sortino_net  regime_vol_spread
A_control     0.0    0.258483 0.274454 0.541011  0.454412        0.481855    0.150000     7.744737  8.230769   0.005421    0.253062    0.521260     -0.153711     -0.173841      0.079245     1.805300           0.042437
A_control     1.0    0.257276 0.248166 0.593457  0.409508        0.419956    0.150000     8.966558 10.000000   0.004725    0.252552    0.574420     -0.151868     -0.195181      0.069621     2.047524           0.081494
A_control     2.0    0.249488 0.201036 0.693844  0.315317        0.315063    0.131055    12.100925 13.153846   0.003544    0.245943    0.676213     -0.148873     -0.179640      0.064315     2.113697           0.062913
A_control     4.0    0.193696 0.153115 0.546626  0.202952        0.213818    0.104527    13.376373 13.923077   0.002405    0.191291    0.530916     -0.134075     -0.156557      0.057611     1.411019           0.071288
A_control     6.0    0.177965 0.143557 0.473438  0.148186        0.168312    0.119183    13.099141 13.846154   0.001894    0.176072    0.460248     -0.135804     -0.154659      0.059151     1.117001           0.106764
A_control     8.0    0.161416 0.143749 0.357679  0.143988        0.170735    0.127293    12.768106 13.615385   0.001921    0.159495    0.344317     -0.150442     -0.167402      0.065104     0.760247           0.116582
A_control    12.0    0.160012 0.145862 0.342875  0.089036        0.127277    0.134888    12.452178 13.384615   0.001432    0.158581    0.333058     -0.159574     -0.170775      0.068758     0.706545           0.119540
A_control    16.0    0.157018 0.144010 0.326492  0.085354        0.124595    0.136573    12.539522 13.538462   0.001402    0.155616    0.316758     -0.159418     -0.170229      0.067840     0.672416           0.122911
 B_target     0.0    0.341939 0.232880 0.995962  0.609421        0.621049    0.150000     8.031807  8.692308   0.006987    0.334952    0.965960     -0.100836     -0.100836      0.041950     5.362327           0.104299
 B_target     1.0    0.342417 0.229545 1.012513  0.579745        0.591908    0.150000     8.260120  9.000000   0.006659    0.335758    0.983504     -0.101527     -0.101527      0.042238     5.344962           0.092517
 B_target     2.0    0.337182 0.233252 0.973974  0.526080        0.530712    0.148681     9.577100 10.692308   0.005971    0.331211    0.948378     -0.106771     -0.106771      0.044420     4.980042           0.074277
 B_target     4.0    0.295040 0.216006 0.856642  0.432445        0.428442    0.137175    11.636912 12.769231   0.004820    0.290220    0.834328     -0.132208     -0.132208      0.055002     3.276621           0.037120
 B_target     6.0    0.260626 0.208362 0.722905  0.362778        0.359948    0.118665    12.600974 13.461538   0.004049    0.256576    0.703471     -0.142666     -0.142666      0.059515     2.462840           0.080986
 B_target     8.0    0.237515 0.193709 0.658281  0.299513        0.308603    0.115905    12.928903 13.769231   0.003472    0.234043    0.640358     -0.147637     -0.147637      0.061602     2.013620           0.086873
 B_target    12.0    0.228619 0.179866 0.659484  0.217769        0.236809    0.122675    13.069269 14.000000   0.002664    0.225955    0.644672     -0.154785     -0.154785      0.064607     1.794780           0.092084
 B_target    16.0    0.199208 0.164210 0.543255  0.162692        0.186221    0.128897    12.920865 13.923077   0.002095    0.197113    0.530497     -0.154747     -0.159807      0.065390     1.332206           0.117616
```

## Cost sensitivity

Optimal `s` per arm across transaction-cost assumptions -- shows whether
the recommendation is stable or hinges on the assumed cost.

```
 cost  A_control_best_s  A_control_sharpe  B_target_best_s  B_target_sharpe
0.001               2.0          0.690318              1.0         1.006711
0.003               2.0          0.683266              1.0         0.995108
0.005               2.0          0.676213              1.0         0.983504
0.010               2.0          0.658582              1.0         0.954495
0.020               2.0          0.623320              1.0         0.896476
```

---
Committed here rather than left in `experiments/results/`, which
`.gitignore:31` excludes -- the reason the original architecture-study
numbers were unrecoverable.
