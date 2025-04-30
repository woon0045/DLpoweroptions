**Deep Learning Approaches to Mean-variance Portfolio Hedging
**

This repository contains the implementation of the deep learning algorithm presented in the paper "_Deep learning approaches to mean-variance portfolio hedging_".
 
_Credit: Codes were adapted from the research by Agram et al. (2024), available on GitHub at janrems/DeepLearningQHedging
_
Modifications were made to evaluate LSTM-based hedging strategies for power call options under three market models: 
the Black-Scholes (BS) complete market, the Merton jump-diffusion model, and the Mixed Merton model. 

The BS model is used as the benchmark, hence one must run the file 'BlackScholes_PowerOptions.py' first to obtain the benchmark values.
