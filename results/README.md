# Results, Figures and Tables

## Raw result Files

Raw result files, as outputed by the run-scripts "eval__*" (see here: [CASMI (2016)](/msmsrt_scorer/experiments/CASMI_2016) and [EA Massbank](/msmsrt_scorer/experiments/EA_Massbank)), are provided for selected [package versions](/CHANGELOG.md). The directories corresponding to both datasets contain the raw result files for different settings (TFG = Tree Factor Graph):

| Directory | MS2 Base Scorer | [Modes](/msmsrt_scorer/experiments/CASMI_2016/eval__TFG.py#L155) | Note |
| --- | --- | --- | --- |
| results__MetFrag22 | [MetFrag 2.2](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9) | application | Retention time weight D was determined using grid-serach (see Section 3.4 and 4.2.2). |
| results__TFG__gridsearch | Our, [Bach et al. (2018)](https://academic.oup.com/bioinformatics/article/34/17/i875/5093227) | application, development, missing_ms2 | Retention order weight D _and_ Sigmoid parameter k where determined using grid-serach. |
| results__TFG__platt | Our | application, development, missing_ms2 | Retention order weight D was determined using grid-search and Sigmoid parameter k using Platt's method (see Section 3.4) | 

## Tables and Figures

The IPython notebooks ("make_*.ipynb") produce the figures and tables of the paper directly from the raw data. The raw data package version chosen in the notebooks by setting the ```result_version``` variable.   

### Install Virtual Environment for JupyterLab

We developped the notebooks using JupyterLab (>= 2.1). You can install JupyterLab as described [here](https://github.com/jupyterlab/jupyterlab). The notebooks, however, should run with earlier versions as well. If you [followed the instructions](https://github.com/aalto-ics-kepaco/msms_rt_score_integration#install-into-a-virtual-environment) to install this package into a virtual environment, you need to add this environment as kernel to JupyterLab, so that you can re-run the notebools: 

1. Open a terminal and activate the virtual environment (assuming your are in the base-directory of this repository): ```source msmsrt_scorer_venv/bin/activate```
2. Install the "ipykernel" package (only needed for the first time): ```pip install ipykernel```
3. Run: ```python -m ipykernel install --user --name=msmsrt_scorer_venv```
 - The command should print the following: ```Installed kernelspec myenv in /home/USER/.local/share/jupyter/kernels/msmsrt_scorer_venv```
4. When you run the JupyterLab start the desired notebook and select "Change Kernel..." in the "Kernel" drop down menu. 

This description follows the instructions given [here](https://janakiev.com/blog/jupyter-virtual-envs/).

### Notebook Overview

| Notebook | Description | 
| --- | --- |
| [make_figures.ipynb](./make_figures__v0.2.0.ipynb) | Produces all result figures in the paper: 2 and 3 |
| [make_tables.ipynb](./make_tables__v0.2.0.ipynb) | Produces all result tables in the paper: 2, 3, 4, 5 and S1. |
