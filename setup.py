from setuptools import setup

setup(
    name="annchor",
    version="1.1.0",
    author="Jonathan H",
    packages=["annchor"],
    description="Fast k-NN graph construction for slow metrics",
    install_requires=[
        "joblib>=1.0.1",
        "numpy<2,>=1.18",
        "numba>=0.55.1",
        "python-Levenshtein>=0.12.2",
        "pynndescent>=0.5.4",
        "scipy>=1.7.0",
        "scikit-learn>=0.0",
        "tqdm>=4.61.2",
    ],
    package_data={"annchor": ["data/*.npz", "data/*.gz"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
