# Utility Functions 

Implementation utility functions used to run the experiments with our MS2 + RT score integration
framework.

## Requirements

The following packages are required in the specified minimum version:

* numpy >= 1.17
* scipy >= 1.3
* sklearn >= 0.22
* pandas >= 0.25.3
* [gm_solver](src/gm_solver) (shipped with this repository)


## Install

To install the package, simply run: ```python setup.py install```.

## Data Handling: ```data_utils.py```

Implementation of the functionality needed to load the data from the [local SQLite
database](data) and provide the needed data structures to our framework.  

- Loading spectra information ([here](msmsrt_scorer/data_utils.py#L179) and [here](msmsrt_scorer/data_utils.py#281))
- Loading [candidate information](msmsrt_scorer/data_utils.py#L217) 
- Loading [MS2-scores](msmsrt_scorer/data_utils.py#L217) and [RankSVM preferences scores](msmsrt_scorer/data_utils.py#L202)
- Implementation of the edge-potential-functions
 
If you want to use our framework without a local SQLite database, you can 
start by replacing the SQLite statements by, e.g., accesses to CSV-files. Those
files simply need to provide the same information. 

## Metabolite Identification Performance Evaluation: ```evaluation_tools.py```

Functionality to calculate top-k ranking performance measures used in our paper. Furthermore, 
implementations needed to run the hyper parameter grid search and best parameter selection.

- [Top-k accuracy calculation](msmsrt_scorer/evaluation_tools.py#L270)
- [Grid search](msmsrt_scorer/evaluation_tools.py#L39)
- [Performance evaluation of grid elements](msmsrt_scorer/evaluation_tools.py#L167)