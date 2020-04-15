# Sum- and Max-Product Implementation: ```exact_solvers.py```

Implementation of [Sum-product and Max-product algorithm](gm_solver/exact_solvers.py#L302) 
for tree like Markov random field to calculate the candidate marginals. We 
closely followed the description of [1, p. 334] and [2, p. 383] for the 
implementation. Also the [random spanning tree sampling](gm_solver/exact_solvers.py#L738) 
is implemented in this file. 

# References

* [1]: [MacKay, D. J., "Information theory, inference and learning algorithms", Cambridge university press (2005)](http://www.inference.org.uk/mackay/itila/)
* [2]: [Bishop, C., "Pattern Recognition and Machine Learning", Springer New York (2006)](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)