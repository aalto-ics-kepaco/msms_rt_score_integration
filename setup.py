from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('msmsrt_scorer/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="msmsrt_scorer",
    version=main_ns["__version__"],
    license="MIT",
    packages=find_packages(exclude=["results*", "tests", "examples", "*.ipynb"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.3",
        "scikit-learn>=0.22",
        "joblib>=0.14.0",
        "pandas>=0.25.3",
        "matplotlib>=3.1",
        "seaborn>=0.9.0",
        "networkx>=2.4",
        "setuptools>=46.1"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Implementation of an MS + RT score integration framework. The details of the framework are described"
                "in 'Probabilistic Framework for Integration of Mass Spectrum and Retention Time Information in "
                "Metabolite Identification' by Eric Bach, Simon Rogers, John Williamson and Juho Rousu (2020).",
    url="https://github.com/aalto-ics-kepaco/msms_rt_score_integration",
)
