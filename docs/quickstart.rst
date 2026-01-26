Quickstart
==========

This guide shows basic usage of the stochastic_dynamics package.

Generating Signals
------------------

Generate a Lorenz attractor:

.. code-block:: python

   from stochastic_dynamics import LorenzGenerator

   gen = LorenzGenerator(sigma=10, rho=28, beta=8/3)
   data = gen(n_steps=10000, dt=0.01)  # (10000, 3)
   
   # Access time vector
   t = gen.time

Generate a time-varying AR process:

.. code-block:: python

   from stochastic_dynamics import TVARGenerator

   gen = TVARGenerator(order=2, modulation='sine', mod_freq=0.1)
   data = gen(n_steps=5000, dt=1.0)

Using Analysis Tools
--------------------

Compute power spectral density:

.. code-block:: python

   from stochastic_dynamics import PSDTool

   psd = PSDTool(nperseg=256)
   freqs, power = psd(signal, fs=500)

Compute delay embedding:

.. code-block:: python

   from stochastic_dynamics import DelayEmbedder

   emb = DelayEmbedder(m=3, tau=15)
   embedded = emb(signal)  # (n_embedded, 3)

Processing Real Data
--------------------

Process mouse hippocampal LFP:

.. code-block:: python

   from stochastic_dynamics import MouseLFPPipeline

   pipeline = MouseLFPPipeline(fs=20000, ripple_band=(80, 140))
   processed = pipeline(raw_lfp)
   ripples = pipeline.detect_ripples(threshold_sd=3.0)

Process monkey visual cortex LFP:

.. code-block:: python

   from stochastic_dynamics import MonkeyLFPPipeline

   pipeline = MonkeyLFPPipeline(fs=500, notch_freqs=(60, 120))
   processed = pipeline(raw_lfp)
   freqs, psd = pipeline.compute_psd()
