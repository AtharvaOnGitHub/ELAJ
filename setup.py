from setuptools import setup, find_packages

setup(
    name="endo_model",
    version="0.1.0",
    description="ELAJ: Endometriosis Latent space for Analysis and Judicious review",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "anndata>=0.9.0",
        "scanpy>=1.9.0",
        "pandas>=1.5.0",
        "h5py>=3.8.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
