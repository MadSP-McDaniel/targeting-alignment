from setuptools import find_packages, setup

# Core dependencies required for basic functionality
CORE_DEPS = [
    "numpy",
    "pandas",
    "transformers>=4.40",
    "accelerate",
    "torch",
    "typing",
    "fschat",
    "sentencepiece",
    "protobuf",
    "argparse",
    "scikit-learn",
    "statsmodels",
]

# Optional dependencies for visualization and interactive usage
VIZ_DEPS = ["matplotlib", "seaborn"]

# Hardware-specific Polars dependencies
POLARS_DEPS = {
    "polars": ["polars"],  # Default Polars
    "polars-lts": ["polars-lts-cpu"],  # LTS CPU version
}

setup(
    name="clfextract",
    version="0.1.0",
    description="A framework to extract safety classifiers of LLMs.",
    author="Jean-Charles Noirot Ferrand",
    author_email="jcnf@cs.wisc.edu",
    python_requires=">=3.10",
    url="https://github.com/jcnf0/targeting-alignment",
    packages=find_packages(),
    install_requires=CORE_DEPS,
    extras_require={
        # Individual feature sets
        "viz": VIZ_DEPS,
        "polars": POLARS_DEPS["polars"],
        "polars-lts": POLARS_DEPS["polars-lts"],
        # Convenience combinations
        "full": VIZ_DEPS + POLARS_DEPS["polars"],
        "full-lts": VIZ_DEPS + POLARS_DEPS["polars-lts"],
        # Development dependencies
        "dev": [
            "pytest",
            "debugpy",
            "black",
            "isort",
            "flake8",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
