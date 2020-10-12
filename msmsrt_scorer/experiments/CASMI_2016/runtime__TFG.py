####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import argparse
import numpy as np
import os
import gzip
import pickle
import sys
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit, ParameterGrid

from msmsrt_scorer.lib.data_utils import prepare_candidate_set_MetFrag, prepare_candidate_set_IOKR
from msmsrt_scorer.lib.evaluation_tools import get_topk_performance_from_scores, evaluate_parameter_grid, get_marginals
from msmsrt_scorer.lib.evaluation_tools import run_parameter_grid, get_top20AUC

from msmsrt_scorer.experiments.CASMI_2016.eval__TFG import load_data, load_platt_k

# Use fixed values for some of the parameters (corresponding to the ones used in the paper)
PARAM_SELECTION_MEASURE = "topk_auc"
D_VALUE_GRID = []
ORDER_PROB_K_GRID = "platt"
MARGIN_TYPE = "max"
MAKE_ORDER_PROB = "sigmoid"
NORM_ORDER_SCORES = False
TREE_METHOD = "random"
N_RANDOM_TREES = 128
PARTICIPANT = "MetFrag_2.4.5__8afe4a14"  # MS2-scorer are pre-computed, i.e. the method doesn't influence the runtime
PREF_MODEL = "c6d6f521"  # RankSVM preference scores are pre-computed, i.e. the model doesn't influence the runtime
SORT_CANDIDATES_BY_MS2_SCORE = False
RESTRICT_CANDIDATES_TO_CORRECT_MF = False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--ion_mode", type=str, default="positive",
                            help="Load spectra only of the specified ionization mode. In the paper we always analyzed "
                                 "negative and positive mode spectra separately.")

    arg_parser.add_argument("--mode", type=str, default="debug_runtime", choices=["debug_runtime", "runtime"])

    arg_parser.add_argument("--database_fn", type=str, help="Path to the score SQLite DB.")

    arg_parser.add_argument("--base_odir", type=str, default="results__TFG",
                            help="Base directory to store the results and output files.")

    arg_parser.add_argument("--n_jobs", type=int, default=4,
                            help="Number of jobs used to parallelize the score-integration on the spanning-tree"
                                 "ensembles.")

    args = arg_parser.parse_args()
    print(str(args))

    # Define output directory
    if args.tree_method == "random":
        _tree_description = "random__n_trees=%d" % N_RANDOM_TREES
        n_trees = N_RANDOM_TREES
    else:
        raise ValueError("Invalid tree method: '%s'" % TREE_METHOD)

    _tmp = ["tree_method=%s" % _tree_description,
            "make_order_prob=%s" % MAKE_ORDER_PROB,
            "param_selection_measure=%s" % PARAM_SELECTION_MEASURE,
            "norm_order_scores=%d" % NORM_ORDER_SCORES,
            "mtype=%s" % MARGIN_TYPE,
            "crcmf=%d" % RESTRICT_CANDIDATES_TO_CORRECT_MF]

    odir = os.path.join(args.base_odir, args.mode,
                        "__".join(_tmp),
                        "__".join(["ion_mode=%s" % args.ion_mode,
                                   "participant=%s" % PARTICIPANT,
                                   "max_n_cand=%.0f" % args.max_n_cand,
                                   "pref_model=%s" % PREF_MODEL,
                                   "sort_candidates_by_ms2_score=%d" % args.sort_candidates_by_ms2_score]))
    os.makedirs(odir, exist_ok=True)