# Python-MAGI
A TensorFlow Probability-Powered Upgraded Implementation of [Manifold-Constrained Gaussian Process Inference (MAGI)](https://www.pnas.org/doi/10.1073/pnas.2020397118).

This package is an upgraded version of the MAGI algorithm developed by [Yang et al. (2020)](https://www.pnas.org/doi/10.1073/pnas.2020397118) for solving the ordinary differential equation (ODE) dynamical system inverse problem. Compared to the [original MAGI implementation](https://www.jstatsoft.org/article/view/v109i04), this version is based in TensorFlow Probability and includes higher-performance routines for completely-missing components and GPU-accelerated MCMC sampling. The package is designed to be essentially hyperparameter-tuning-free, as it uses Hoffman & Gelman's [No U-Turn Sampler](https://sites.stat.columbia.edu/gelman/research/published/nuts.pdf) with Dual-Averaging Stepsize Adaptation, all natively-implemented in TensorFlow Probability.

The main `MAGI` class was written to roughly mirror `scikit-learn` grammar for user-friendliness, with some minor modifications. Please see our [vignette](https://github.com/skbwu/Python-MAGI/blob/main/vignette.ipynb) for a quick start guide.

**High-Level Description of Core Functions:** please see the complete source code at `magi.py` for full details.
- `__init__`: constructor where the governing ODE equations `f_vec`, (potentially partially-observed) timesteps `ts_obs` and noisy observations `X_obs`, and computational settings (e.g., number of components in system, bandmatrix approximations) are specified to create a `MAGI` object.
- `initial_fit`: fits the initial Matern kernel hyperparameters for the ODE inverse problem. Can take user-specified subsets of hyperparameters, too, and automatically accounts for missing observations and/or entirely-missing components in the provided noisy observed data.
- `predict`: after receiving and/or fitting Matern kernel hyperparameters to the provided data, returns samples from the posterior distribution of the entire system along with diagnostics.

Please contact [skylerw@stanford.edu](skylerw@stanford.edu) if you have any comments, suggestions, or concerns. 
