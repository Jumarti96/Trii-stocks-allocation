# Backtest: model vs naive alternatives

Walk-forward, cadences [12, 24], baseline `ew_all`, arch `B`, loss `rank_ic`.

All comparisons are **paired** per rebalance window: the books are ~98%
correlated, so differencing cancels the market move and leaves the
strategy effect. `vs_*` columns are against the baseline.

## Results

```
 cadence    strategy  n_windows     cagr  ann_vol   sharpe  net_cagr  net_sharpe  max_drawdown  worst_window  turnover    n_held  random_pct  vs_mean_diff      vs_t     vs_p  vs_wins  vs_n_for_80_power
      12    model_s1         26 0.302961 0.196006 0.984466  0.291830    0.927679     -0.164895     -0.164895  0.494696  9.153846    0.587077      0.018765  1.626457 0.116393     16.0          77.055623
      12    model_s2         26 0.292838 0.197899 0.923896  0.282700    0.872665     -0.193588     -0.167506  0.450602 10.730769    0.588846      0.017036  1.510652 0.143415     17.0          89.322382
      12    model_s0         26 0.293844 0.199308 0.922412  0.282251    0.864246     -0.168571     -0.168571  0.515243  8.692308    0.568615      0.017259  1.421506 0.167525     15.0         100.877008
      12    model_s4         26 0.278794 0.188374 0.896056  0.270775    0.853487     -0.156708     -0.142510  0.356395 12.653846    0.573692      0.014056  1.314888 0.200482     14.0         117.899405
      12      ew_all         26 0.214087 0.128012 0.813102  0.212939    0.804133     -0.160686     -0.122877  0.051027 80.000000    0.532923           NaN       NaN      NaN      NaN                NaN
      12 inverse_vol         26 0.201525 0.123274 0.742450  0.200414    0.733435     -0.162618     -0.118639  0.049390 80.000000    0.508231     -0.002539 -3.556680 0.001531      6.0          16.113870
      12    model_s6         26 0.238323 0.171317 0.749040  0.231441    0.708867     -0.142929     -0.141396  0.305876 13.423077    0.528077      0.005930  0.671185 0.508257     13.0         452.485693
      12     ew_topn         26 0.248498 0.189742 0.729928  0.238232    0.675824     -0.184310     -0.184310  0.456253  8.692308    0.550000      0.008596  0.748047 0.461411     12.0         364.277038
      12    model_s8         26 0.210162 0.162483 0.616445  0.204596    0.582192     -0.184502     -0.153367  0.247362 13.615385    0.495077      0.000313  0.038696 0.969440     12.0      136127.891066
      12    momentum         26 0.235529 0.197366 0.636020  0.221456    0.564720     -0.239755     -0.166398  0.625425  8.692308    0.547231      0.006433  0.513930 0.611814     13.0         771.759575
      12         gmv         26 0.155141 0.105216 0.429032  0.153900    0.417238     -0.160654     -0.105886  0.055153 55.307692    0.422000     -0.012059 -2.857101 0.008488      7.0          24.971136
      12       sp500         26 0.155958 0.154328 0.297796  0.155958    0.297796     -0.225197     -0.134160  0.000000  1.000000         NaN     -0.010504 -0.886166 0.383973     10.0         259.572659
      24    model_s0         13 0.360477 0.213003 1.175929  0.353286    1.142170     -0.042885     -0.042885  0.639181  8.461538    0.737846      0.059300  2.665343 0.020587     11.0          14.346741
      24     ew_topn         13 0.343792 0.200457 1.166292  0.336942    1.132123     -0.070693     -0.070693  0.608831  8.461538    0.686308      0.052305  1.936851 0.076663      9.0          27.168585
      24    model_s1         13 0.361873 0.216547 1.163135  0.354976    1.131286     -0.063741     -0.063741  0.613034  8.923077    0.747077      0.060170  2.723602 0.018484     12.0          13.739538
      24    model_s2         13 0.357558 0.221489 1.117700  0.351361    1.089720     -0.090402     -0.090402  0.550881 10.692308    0.734462      0.059072  2.800227 0.016037     11.0          12.997890
      24    model_s4         13 0.336433 0.215709 1.049712  0.331611    1.027362     -0.072071     -0.072071  0.428544 12.923077    0.689692      0.050723  2.358651 0.036140     10.0          18.320275
      24    model_s6         13 0.287808 0.202541 0.877886  0.283790    0.858048     -0.100765     -0.100765  0.357168 13.615385    0.632615      0.031467  1.730065 0.109229      8.0          34.051365
      24    model_s8         13 0.255801 0.199645 0.730303  0.252604    0.714285     -0.152255     -0.152255  0.284253 13.692308    0.578000      0.019220  1.146889 0.273781      7.0          77.484816
      24      ew_all         13 0.213938 0.159804 0.650405  0.213104    0.645188     -0.147058     -0.147058  0.074112 80.000000    0.542154           NaN       NaN      NaN      NaN                NaN
      24 inverse_vol         13 0.202118 0.152601 0.603653  0.201304    0.598320     -0.142736     -0.142736  0.072350 80.000000    0.516000     -0.005189 -3.522162 0.004207      3.0           8.215630
      24    momentum         13 0.221584 0.170801 0.653297  0.211941    0.596838     -0.165016     -0.165016  0.857176  8.461538    0.539385      0.003617  0.175766 0.863409      6.0        3299.042589
      24         gmv         13 0.156769 0.123492 0.378723  0.155839    0.371193     -0.128269     -0.128269  0.082662 55.384615    0.407692     -0.025186 -2.634929 0.021777      3.0          14.679844
      24       sp500         13 0.155958 0.169958 0.270409  0.155958    0.270409     -0.140667     -0.128319  0.000000  1.000000         NaN     -0.022766 -0.833408 0.420894      7.0         146.738497
```

## How to read this

- **`random_pct`** is the headline for *is the model useful*: the mean
  percentile of the strategy within a distribution of random books of the
  same size. 0.5 means indistinguishable from picking at random.
- **`gmv`** uses only the covariance and ignores the forecast. If it
  matches the model, the transformer contributes nothing and the value is
  in the Ledoit-Wolf estimate.
- **`n_for_80_power`** is how many windows would be needed to call the
  paired difference significant at the observed effect size.

## Limitations

- **Survivorship.** The universe is today's Trii catalogue filtered to full
  history, so everything that delisted is absent. This flatters a
  stock-picking strategy more than it flatters equal-weight.
- **One regime.** Every window is post-2020 and mostly rising.
- **No FX.** `01_download.py` takes `pct_change()` on native-currency
  prices; COP, USD, CLP and CHF are summed as one unit. The S&P 500 row is
  approximate, and US holdings' true COP returns were higher than shown.
- **Power is bounded by calendar span**, not window count: shorter cadence
  gives more windows but proportionally less signal in each.

---
Committed here because `.gitignore:31` excludes `experiments/results/`.
