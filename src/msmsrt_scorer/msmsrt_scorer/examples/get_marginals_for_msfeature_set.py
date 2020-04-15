####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
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
import sqlite3
import scipy.stats
import numpy as np

from sklearn.model_selection import ShuffleSplit

from gm_solver.exact_solvers import RandomTreeFactorGraph

from msmsrt_scorer.data_utils import sigmoid, prepare_candidate_set_IOKR, load_dataset_CASMI, hinge_sigmoid
from msmsrt_scorer.evaluation_tools import get_topk_performance_casmi2016

import logging
logging.basicConfig(format='%(name)s: %(message)s', level=logging.WARNING)
LOGGER = logging.getLogger("Example 01")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--database_fn", type=str,
                            default="/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db")
    args = arg_parser.parse_args()

    LOGGER.setLevel(level=logging.DEBUG)

    # Path to the CASMI score database
    DBFN = args.database_fn
    DBURL = "file:" + DBFN + "?mode=ro"

    # Ionization mode to consider
    MODE = "positive"

    # Load MS2 scores of specified method
    METHOD = "MetFrag_2.4.5__8afe4a14"

    # Preference score model
    PREFMODEL = "c6d6f521"

    with sqlite3.connect(DBURL, uri=True) as db:
        challenges, candidates = load_dataset_CASMI(db, MODE, METHOD, PREFMODEL, max_n_cand=np.inf,
                                                    sort_candidates_by_ms2_score=False)

    # Get random candidate set subset
    _, subs = next(ShuffleSplit(1, test_size=75, random_state=2020).split(candidates))

    candidates = prepare_candidate_set_IOKR(challenges, candidates, subs, random_state=99, n_ms2=75)

    def f1(pref_diff, **kwargs):
        return scipy.stats.norm.pdf(pref_diff, loc=kwargs["loc"], scale=1)

    def f2(pref_diff, **kwargs):
        return sigmoid(pref_diff, k=1)

    def f3(pref_diff, **kwargs):
        return sigmoid(pref_diff, x_0=kwargs["loc"])

    def f4(pref_deff, **kwargs):
        return hinge_sigmoid(pref_deff)

    TFG = RandomTreeFactorGraph(candidates, make_order_probs=f2, D=0.1, random_state=332).max_product()
    marg = TFG.get_max_marginals()
    print(TFG)

    _, acc = get_topk_performance_casmi2016(candidates, None)
    print(acc[0], acc[4], acc[9], acc[19])
    _, acc = get_topk_performance_casmi2016(candidates, marg)
    print(acc[0], acc[4], acc[9], acc[19])
