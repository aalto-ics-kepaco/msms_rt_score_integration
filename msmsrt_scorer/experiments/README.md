# Produce Raw Results

Scripts to produce the raw results that can be used to [create plots and tables](/results/). For both datasets ([CASMI 2016](/msmsrt_scorer/experiments/CASMI_2016) and [EA (Massbank)](/msmsrt_scorer/experiments/EA_Massbank)) there are three scripts:

| Script | MS2 Base Scorer | Note | Reference in the Paper |
| --- | --- | --- | --- |
| eval__MetFrag22 | [MetFrag 2.2](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9) | Evaluation of metabolite identification performance | Section 4.3.1, Table 3 | 
| eval__TFG | Our, [Bach et al. (2018)](https://academic.oup.com/bioinformatics/article/34/17/i875/5093227) | Evaluation of metabolite identification performance | Section 4.3.1, Table 3 | 
| | | Inspect parameters of our framework, e.g. margin type or number of spanning trees | Section 4.2.* | 
| | | Inspect hyper-parameter estimation | Section 4.2.2 | 
| eval__TGF__missing_MS2 | Our | Evaluation of score integration framework for missing tandem mass spectra (MS2) | Section 4.4 | 

## Re-run Experiments

Here, we will describe how the experiments can be re-run on the example of [```eval__TFG.py``` (EA Massbank)](/msmsrt_scorer/experiments/EA_Massbank/eval__TFG.py#L82). Assuming you have installed the [```nmsmsrt_scorer``` package and, if needed, activated the virtual environment](https://github.com/aalto-ics-kepaco/msms_rt_score_integration#install-into-a-virtual-environment), you can run the evaluation script as follows:
```bash
python EA_Massbank/eval__TFG.py \
      --mode=EVALUATION_MODE \
      --D_value_grid 0.001 0.005 0.01 0.05 0.1 0.15 0.25 0.35 0.5 \
      --make_order_prob=EDGE_POTENTIAL_FUNCTION \
      --order_prob_k_grid platt \
      --margin_type=MARGIN_TYPE \
      --n_random_trees=NUMBER_OF_RANDOM_TREES_FOR_APPROXIMATION \
      --n_samples=NUMBER_OF_RANDOM_TEST_TRAINING_SETS \
      --ion_mode=IONIZATION_MODE \
      --max_n_ms2=NUMBER_OF_MS2_FOR_TEST \
      --database_fn=SCORE_DB_FN \
      --base_odir=BASE_OUTPÙT_DIRECTORY \
```

### Detailed description of selected parameters 

A description of all parameters, can be found in the [```__main___```](/msmsrt_scorer/experiments/EA_Massbank/eval__TFG.py#L82) of the "eval__" script files. Some selected parameters will be explained here: 

```--mode```
| EVALUATION_MODE [[1](/msmsrt_scorer/experiments/EA_Massbank/eval__TFG.py#L174), [2](/msmsrt_scorer/experiments/EA_Massbank/eval__TFG__missing_MS2.py#L151)] | Description |
| --- | --- |
| application | Results to Evaluate the performance on the test sets in the application setting. |
| development | Performance evaluation of training _and_ test set for each hyper parameter grid value |
| missing_ms2 | Performance evaluation for the mssing MS2 experiment |

```--D_value_grid```

Grid used to search for the best retention order weight (see Section 2.2.4 and 3.4). We use ```[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5]``` in our experiments.

```--order_prob_k_grid```

Grid used to search for the best sigmoid slope parameter when using EDGE_POTENTIAL_FUNCTION=sigmoid or EDGE_POTENTIAL_FUNCTION=hinge_sigmoid (see Section 2.2.3 and 3.4). As grid we use ```[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 10.0``` for the Hinge-Sigmoid. Our experiments shows that for Sigmoid we can use Platt's ("platt") method to determine the optimal value for k (see Section 4.2.2).

```--ìon_mode``` and ```--max_n_ms2```

These two parameter controll which ionization mode should be evaluation (negative or positive) and how many MS-features are used to calculate the test accuracy. The following settings are available (see Section 3.1):

| Dataset | IONIZATION_MODE | NUMBER_OF_MS2_FOR_TEST | 
| --- | --- | --- |
| CASMI (2016) | positive | 75 | 
| | negative | 50 | 
| EA (Massbank) | positive | 100 | 
| | negative | 65 | 


```--n_smaples```

Number of (training, test)-set samples. In our experiments we use NUMBER_OF_RANDOM_TEST_TRAINING_SETS=50 (CASMI, EA (negative)) and NUMBER_OF_RANDOM_TEST_TRAINING_SETS=100 (EA (positive))

```--db_fn```

Path to the [SQLite DB](/data/). 

```--base_odir```

Path to the output directory storing the [raw results](/results/EA_Massbank/results__TFG__platt/). The output directory will sub-directories separating the results resulting from different parameter settings. 


### Example: EA (Massbank) positive, Results for Table 3

```bash
python EA_Massbank/eval__TFG.py \
      --mode=development_debug \
      --D_value_grid 0.001 0.005 0.01 0.05 0.1 0.15 0.25 0.35 0.5 \
      --make_order_prob=EDGE_POTENTIAL_FUNCTION \
      --order_prob_k_grid platt \
      --margin_type=MARGIN_TYPE \
      --n_random_trees=NUMBER_OF_RANDOM_TREES_FOR_APPROXIMATION \
      --n_samples=NUMBER_OF_RANDOM_TEST_TRAINING_SETS \
      --ion_mode=IONIZATION_MODE \
      --max_n_ms2=NUMBER_OF_MS2_FOR_TEST \
      --database_fn=SCORE_DB_FN \
      --base_odir=BASE_OUTPÙT_DIRECTORY \
```
