from setuptools import setup, find_packages

setup(
    name="gm_solver",
    version="0.1",
    license="MIT",
    packages=find_packages(exclude=["*tests"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.3",
        "scikit-learn>=0.22",
        "networkx>=2.4",
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Solver for the tree approximated graphical models.",
    url="https://github.com/aalto-ics-kepaco/msms_rt_score_integration",
)
