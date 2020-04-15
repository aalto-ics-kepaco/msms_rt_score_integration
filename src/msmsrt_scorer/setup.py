from setuptools import setup, find_packages

setup(
    name="msmsrt_scorer",
    version="0.1",
    license="MIT",
    packages=find_packages(exclude=["*tests"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.3",
        "scikit-learn>=0.22",
        "pandas>=0.25.3",
        "joblib>=0.14.0",

        # Note: Our graphical model solver is not indexed online. Please install the
        #       package first from this repository: "python ../gm_solver/setup.py install".
        "gm_solver"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Utility functions for the MS2 + RT score integration method.",
    url="https://github.com/aalto-ics-kepaco/msms_rt_score_integration",
)
