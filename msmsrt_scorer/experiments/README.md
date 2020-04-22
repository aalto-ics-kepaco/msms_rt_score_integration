# Produce Raw Results

Scripts to produce the raw results that can be used to [create plots and tables](/results/). For both datasets ([CASMI 2016](/msmsrt_scorer/experiments/CASMI_2016) and [EA (Massbank)](/msmsrt_scorer/experiments/EA_Massbank)) there are three scripts:

| Script | MS2 Base Scorer | Note | Reference in the Paper |
| --- | --- | --- | --- |
| eval__MetFrag22 | [MetFrag 2.2](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9) | Evaluation of metabolite identification performance | Section 4.3.1, Table 3 | 
| eval__TFG | Our, [Bach et al. (2018)](https://academic.oup.com/bioinformatics/article/34/17/i875/5093227) | Evaluation of metabolite identification performance | Section 4.3.1, Table 3 | 
| | | Inspect parameters of our framework, e.g. margin type or number of spanning trees | Section 4.2.* | 
| | | Inspect hyper-parameter estimation | Section 4.2.2 | 
| eval__TGF__missing_MS2 | Our | Evaluation of score integration framework for missing tandem mass spectra (MS2) | Section 4.4 | 

## 


application ... Results to Evaluate the performance on the test sets in the application setting. 
development ... Performance evaluation of training _and_ test set for each hyper parameter grid value
missing_ms2 ... Performance evaluation for the mssing MS2 experiment
