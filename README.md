# *bayessynth:* BSC Models in Python

The bayessynth package is a Python implementation of the Bayesian Synthetic Control (BSC). BSC is a probabilistic method for quantitative social science, developed in [Tuomaala (2019)](https://arxiv.org/abs/1910.06106 "Arxiv preprint")[^fn]. It includes tools to estimate the BSC model with Markov Chain Monte Carlo (MCMC) sampling and to analyze and visualize the results.

## Documentation

Limited documentation for the library is available separately within this git repository.

## Dependencies

Fitting of the BSC model is done using `pymc3`, which itself uses depends on `theano` and `scipy`. Other fundamental dependencies include `numpy`, `pandas`, and `sklearn`, as well as the visualization libraries `matplotlib` and `seaborn`.

## Author

Elias Tuomaala

Website: [www.eliastuomaala.com](www.eliastuomaala.com "Author Homepage")

Email: [mail@eliastuomaala.com](mailto:mail@eliastuomaala.com)

## License

The bayessynth copyright belongs to Elias Tuomaala (2020). It is released under the MIT License.

## Example
```
import numpy as np
import pandas as pd
import bayessynth as bs

data_source, target_country, cutoff_year = 'gdp.csv', 'DEU', 1990
factors = 4
prior_distribution = {
                      'sigma_gamma': 500,
                      'k_mu': 16000,
                      'k_sd': 7000,
                      'k_gamma': 7000,
                      'alpha_sd': 30000,
                      'alpha_mu': 0,
                      'b_mu': 0,
                      'b_sd': 1,
                      'b_gamma': 1,
                      'delta_mu': 0,
                      'delta_sd': 10000
}
data = pd.read_csv(data_source)

bs.fit(data, target_country, cutoff_year, prior_distribution)
trace = bs.read_tracefile(target, data, factors)
result_summary = bs.summarize_ppc(target_country, data, trace, factors)
bs.plot(result_summary, cutoff_year, target_country, output='display')
```

[^fn]: Elias Tuomaala. (2019) "The Bayesian Synthetic Control: Improved Counterfactual Estimation in the Social Sciences through Probabilistic Modeling." [Arxiv Open Access](https://arxiv.org/abs/1910.06106 "The Bayesian Synthetic Control").
