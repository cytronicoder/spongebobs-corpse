### Methods pipeline

1. `io.load.load_batch_data`: load and normalize input table.
2. `processing.preprocessing.validate_input_frame`: type and NaN filtering.
3. `processing.features.aggregate_by_thickness`: grouped summary + propagated uncertainty.
4. `stats.regression.fit_linear` and `stats.regression.fit_powerlaw`: OLS/WLS model fits with CIs.
5. `stats.hypothesis.pairwise_adjacent_comparisons`: adjacent-condition Welch tests.
6. `viz.plots`: standardized publication figures.
7. `io.save`: uniform artifact output and caption sidecars.

### Equations

- Linear model: $y = m x + c$
- Power-law linearized model: $y = a + b \sqrt{h}$
- Combined uncertainty: $u = \sqrt{u_{random}^2 + u_{instrument}^2}$
