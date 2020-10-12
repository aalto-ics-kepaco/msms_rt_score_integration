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
import pandas as pd
import time

from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit, ParameterGrid

from msmsrt_scorer.lib.data_utils import prepare_candidate_set_MetFrag
from msmsrt_scorer.lib.evaluation_tools import evaluate_parameter_grid, get_marginals
from msmsrt_scorer.lib.evaluation_tools import run_parameter_grid

from msmsrt_scorer.experiments.CASMI_2016.eval__TFG import load_data, load_platt_k

# Use fixed values for some of the parameters (corresponding to the ones used in the paper)
PARAM_SELECTION_MEASURE = "topk_auc"
# -- D_VALUE_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5]
ORDER_PROB_K_GRID = "platt"
MARGIN_TYPE = "max"
MAKE_ORDER_PROB = "sigmoid"
NORM_ORDER_SCORES = False
TREE_METHOD = "random"
# -- N_RANDOM_TREES = 128
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

    arg_parser.add_argument("--n_samples", type=int, default=20,
                            help="Number of random dataset sub-samples. Those have been already pre-sampled and are "
                                 "stored in the DB (see Section 3.1)")

    arg_parser.add_argument("--D_value_grid", nargs="+", type=float,
                            help="Grid-values for the retention order weight. (1 - D) * llh(MS) + D * llh(RT)")

    arg_parser.add_argument("--n_random_trees", type=int, default=128,
                            help="Number of random spanning-trees to average the marginal distribution.")

    arg_parser.add_argument("--max_n_cand", type=float, default=np.inf,
                            help="Limit the maximum number of candidates per ms-feature. We use all candidates in the"
                                 "paper. This option is mainly used for development.")

    arg_parser.add_argument("--max_n_ms2_grid", nargs="+", type=int,
                            help="Number of (MS2, RT)-tuples used for hyper-parameter optimization, i.e. D, and used "
                                 "in the score-integration, i.e. actual metabolite identification. Here, are grid is "
                                 "defined, as we run the runtime analysis for multiple number of (MS2, RT)-tuples.")

    args = arg_parser.parse_args()
    print(str(args))

    # Define output directory
    if TREE_METHOD == "random":
        _tree_description = "random__n_trees=%d" % args.n_random_trees
        n_trees = args.n_random_trees
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
                                   "sort_candidates_by_ms2_score=%d" % SORT_CANDIDATES_BY_MS2_SCORE]))
    os.makedirs(odir, exist_ok=True)

    # Load data
    challenges, candidates = load_data(args, participant=PARTICIPANT, prefmodel=PREF_MODEL,
                                       sort_candidates_by_ms2_score=SORT_CANDIDATES_BY_MS2_SCORE,
                                       restrict_candidates_to_correct_mf=RESTRICT_CANDIDATES_TO_CORRECT_MF)

    # Set up hyper-parameter grid
    if ORDER_PROB_K_GRID == "platt":
        _k = [load_platt_k(args, pref_model=PREF_MODEL)]
        print("Platt k: %.5f" % _k[0])
    else:
        raise ValueError("Invalid k-parameter setting.")
    h_param_grid = ParameterGrid({"D": args.D_value_grid, "k": _k})
    print("Number of grid-pairs: %d" % len(h_param_grid))

    runtimes_df = []

    for n_ms2_idx, max_n_ms2 in enumerate(args.max_n_ms2_grid):
        for s, (_, eval_set) in enumerate(ShuffleSplit(n_splits=args.n_samples, test_size=max_n_ms2,
                                                       random_state=29181).split(candidates)):
            print("n_ms2_idx=%d/%d ; rep=%d/%d" % (n_ms2_idx + 1, len(args.max_n_ms2_grid), s + 1, args.n_samples))

            # Prepare the candidate set, i.e. sub-setting, sampling the MS2 (if needed), ...
            if PARTICIPANT == "MetFrag_2.4.5__8afe4a14":
                cnds_eval = prepare_candidate_set_MetFrag(challenges, candidates, eval_set,
                                                          ms2_idc=range(len(eval_set)), verbose=True, normalize=False)
            else:
                raise ValueError("Invalid participant: '%s'" % PARTICIPANT)

            print("\tn_eval=%d" % len(eval_set))

            # Measure the runtime of the grid-search to find the optimal hyper-parameter
            start_time = time.time()
            _ = evaluate_parameter_grid(
                *run_parameter_grid(cnds_eval, h_param_grid, TREE_METHOD, n_trees, args.n_jobs, MAKE_ORDER_PROB,
                                    NORM_ORDER_SCORES, MARGIN_TYPE))
            duration_training = time.time() - start_time
            print("\ttraining time=%.3fs" % duration_training)

            # Measure the runtime of the score-integration given a fixed set of hyper-parameters
            start_time = time.time()
            _ = Parallel(n_jobs=args.n_jobs)(
                delayed(get_marginals)(cnds_eval, D=0.15, k=_k[0], tree_method=TREE_METHOD, rep=rep,
                                       make_order_prob=MAKE_ORDER_PROB, norm_order_scores=NORM_ORDER_SCORES,
                                       margin_type=MARGIN_TYPE)
                for rep in range(n_trees))
            duration_application = time.time() - start_time
            print("\tapplication time=%.3fs" % duration_application, flush=True)

            # Collect statistics
            _n_cand = [_cnd["n_cand"] for _cnd in cnds_eval.values()]
            _n_cand_tot = np.sum(_n_cand)
            _n_cand_med = np.median(_n_cand)
            _n_cand_avg = np.mean(_n_cand)
            _n_cand_max = np.max(_n_cand)

            runtimes_df.append([max_n_ms2, s, _n_cand_tot, _n_cand_med, _n_cand_avg, _n_cand_max, duration_training,
                                duration_application])

    pd.DataFrame(runtimes_df, columns=["n_ms2rt_tuples", "sample", "n_cand_tot", "n_cand_med", "n_cand_avg",
                                       "n_cand_max", "training_time", "scoring_time"]) \
        .to_csv(os.path.join(odir, "runtime__n_jobs=%d.csv" % args.n_jobs), index=False)

