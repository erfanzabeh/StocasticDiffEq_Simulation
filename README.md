# stochastic-dynamics

Recover intrinsic manifold geometry from noisy, autocorrelated time-series. Built with JAX.

Standard manifold methods fail on signals with strong temporal structure—oscillations, 1/f noise, nonstationarity. This library provides tools to extract low-dimensional geometric structure from such data using lag embedding, time-varying autoregressive models, and geometry-aware learning.

## Install

```bash
pip install -e .
```

