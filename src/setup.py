from setuptools import setup, find_packages

setup(
    name="msmsrt_scorer",
    description="Utility functions for the MS2 + RT score integration method.",
    version="0.1",

    packages=find_packages(where="msmsrt_scorer", exclude=["tests"]),

    # Minimum requirements the package was tested with
    install_requirements=[
        "numpy>=1.17",
        "scipy>=1.3",
        "sklearn>=0.22",
        "pandas>=0.25.3"
    ]
)