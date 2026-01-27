from setuptools import setup, find_packages

setup(
    name="stochastic_dynamics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "jax>=0.4",
        "jaxlib>=0.4",
    ],
    python_requires=">=3.10",
)
