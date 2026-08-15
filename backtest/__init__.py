# backtest/__init__.py
"""
Walk-forward experimental harness for the Strategy Switcher Engine.

Modules
-------
config        run specification (universe, period, costs, seed, splits)
data          point-in-time data with asserted truncation
hmm_schedule  rolling HMM refits, each trained strictly before its fold
runner        the walk-forward loop
metrics       performance statistics from the daily equity curve
leakage       tests that the harness cannot see the future
baselines     buy-and-hold and static equal-weight comparisons
ablations     component-removal ladder
statistics    multi-seed aggregation, bootstrap CIs, deflated Sharpe
"""
