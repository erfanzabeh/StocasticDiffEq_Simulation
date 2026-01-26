Installation
============

Requirements
------------

- Python 3.9+
- NumPy
- SciPy
- PyTorch (for deep learning models)
- Matplotlib (for visualization)

Install from source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/your-username/StocasticDiffEq_Simulation.git
   cd StocasticDiffEq_Simulation
   pip install -e .

Install documentation dependencies
----------------------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx-book-theme myst-nb

Then build:

.. code-block:: bash

   cd docs
   make html
