from icepickle import __version__
from setuptools import setup, find_packages

base_packages = ["scikit-learn>=0.24.0", "h5py>=2.10.0"]

dev_packages = [
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "jupyter>=1.0.0",
    "jupyterlab>=0.35.4",
    "mktestdocs>=0.1.0",
    "pre-commit>=2.17.0",
]


setup(
    name="icepickle",
    version="0.0.1",
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
)
