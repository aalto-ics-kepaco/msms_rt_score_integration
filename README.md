# Overview

Scripts used to run the experiments presented in the paper:

__"Probabilistic Framework for Integration of Mass Spectrum and Retention Time Information in Metabolite Identification"__,

_Eric Bach, Simon Rogers, John Williamson and Juho Rousu_, 2020

# Installation

All code was developed and tested in a Linux environment. Windows or MacOS are currently not officially supported. 
However, most of the code and installation procedure probably just works fine for those operating systems as well. 

## Requirements Packages

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

## Install into a Virtual Environment

Clone the repository:

```git clone https://github.com/aalto-ics-kepaco/msms_rt_score_integration.git```

Change to the directory: 

```cd msms_rt_score_integration```

Create a virtual Python environment and activate it:

```virtualenv msmsrt_scorer_venv && source msmsrt_scorer_venv/bin/activate```

Run the setup. All required packages will be fetched as well:

```pip install .```

# Usage

An example how to reproduce the results can be found [here](/msmsrt_scorer/experiments). 

# Citation

To refer the original publication please use: *TODO*
