# Backend of our Framework

Implementation utility functions used to run the experiments with our MS2 + RT score integration
framework. 

## Functions

### Data Handling: ```data_utils.py```

Implementation of the functionality needed to load the data from the [local SQLite
database](/data) and provide the needed data structures to our framework.  

- Loading spectra information ([here](/msmsrt_scorer/lib/data_utils.py#L413) and [here](/msmsrt_scorer/lib/data_utils.py#589))
- Loading [candidate information](/msmsrt_scorer/lib/data_utils.py#L289) 
- Loading [MS2-scores](/msmsrt_scorer/lib/data_utils.py#L289) and [RankSVM preferences scores](/msmsrt_scorer/lib/data_utils.py#L274)
- Implementation of the edge-potential-functions
 
If you want to use our framework without a local SQLite database, you can 
start by replacing the SQLite statements by, e.g., accesses to CSV-files. Those
files simply need to provide the same information. 

### Metabolite Identification Performance Evaluation: ```evaluation_tools.py```

Functionality to calculate top-k ranking performance measures used in our paper. Furthermore, 
implementations needed to run the hyper parameter grid search and best parameter selection.

- [Top-k accuracy calculation](/msmsrt_scorer/lib/evaluation_tools.py#L250)
- [Grid search](/msmsrt_scorer/lib/evaluation_tools.py#L39)
- [Performance evaluation of grid elements](/msmsrt_scorer/lib/evaluation_tools.py#L167)

### Sum- and Max-Product Implementation: ```exact_solvers.py```

Implementation of [Sum-product and Max-product algorithm](/msmsrt_scorer/lib/exact_solvers.py#L329) 
for tree like Markov random field to calculate the candidate marginals. We 
closely followed the description of [1, p. 334] and [2, p. 383] for the 
implementation. Also the [random spanning tree sampling](/msmsrt_scorer/lib/exact_solvers.py#L768) 
is implemented in this file. 

## References

* [1]: [MacKay, D. J., "Information theory, inference and learning algorithms", Cambridge university press (2005)](http://www.inference.org.uk/mackay/itila/)
* [2]: [Bishop, C., "Pattern Recognition and Machine Learning", Springer New York (2006)](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
