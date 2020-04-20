
## Installation

All code was developed and tested in a Linux environment. Windows or MacOS are currently not officially supported. 
However, most of the code and installation procedure probably just works fine for those operating systems as well. 

### Requirements Packages

The code has been developed for **Python >= 3.6** and the following packages are required in their specified minimum 
version:

* numpy >= 1.17
* scipy >= 1.3
* pandas >= 0.25.3
* scikit-learn>=0.22
* joblib >= 0.14
* matplotlib >= 3.1
* seaborn >= 0.9
* networkx >= 2.4
* setuptools >= 46.1

### Install into a Virtual Environment

First create a virtual Python environment and activate it:

```virtualenv msmsrt_scorer_venv && source msmsrt_scorer_venv/bin/activate```

Subsequently you can run the setup. All required packages will be fetched as well:

```python setup.py install```
