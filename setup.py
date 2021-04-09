from setuptools import setup

setup(
    name="annchor",
    version="0.1.0",
    author="Jonathan H",
    packages=["annchor"],
    description="Fast k-NN graph construction for slow metrics",
    install_requires=["numpy", "numba", "sklearn"],
)
