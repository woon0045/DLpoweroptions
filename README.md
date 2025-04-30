# Deep Learning Approaches to Mean-Variance Portfolio Hedging

This repository contains the implementation of the deep learning algorithm presented in the paper  
_**"Deep Learning Approaches to Mean-Variance Portfolio Hedging"**_.

> **Credit**: Code was adapted from the research by Agram et al. (2024), available on GitHub at [janrems/DeepLearningQHedging](https://github.com/janrems/DeepLearningQHedging).

Modifications were made to evaluate LSTM-based hedging strategies for power call options under three market models:

- The Black-Scholes (BS) complete market: `BlackScholes_PowerOptions.py`  
- The Merton jump-diffusion model: `MertonModel_PowerOptions.py`  
- The Mixed Merton model: `MixedMerton_PowerOptions.py` 

The BS model is used as the benchmark. To begin, run the file `BlackScholes_PowerOptions.py` to generate benchmark values.
