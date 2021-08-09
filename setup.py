from setuptools import setup

setup(
    name="annchor",
    version="1.0.0",
    author="Jonathan H",
    packages=["annchor"],
    description="Fast k-NN graph construction for slow metrics",
    install_requires=[
        "joblib>=1.0.1",
        "numpy>=1.21.0",
        "numba>=0.53.1",
        "python-Levenshtein>=0.12.2",
        "pynndescent>=0.5.4",
        "scipy>=1.7.0",
        "sklearn>=0.0",
        "tqdm>=4.61.2",
    ],
    package_data={"annchor": ["data/*.npz", "data/*.gz"]},
)
