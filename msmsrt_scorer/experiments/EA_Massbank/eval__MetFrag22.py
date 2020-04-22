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

# Script to run the experiments presented in Section 4.2 of the paper for the EA (Massbank) datasets using the MetFrag
# 2.2 method (for comparison).

import argparse
import sqlite3
import numpy as np
import os
import gzip
import pickle
import sys
import pandas as pd
import tarfile

# NOTE: Read the README.md to get information on RDKit
from rdkit.Chem.inchi import MolToInchi, MolFromInchi
from rdkit import RDLogger
from collections import OrderedDict
from sklearn.model_selection import ShuffleSplit, ParameterGrid

from msmsrt_scorer.lib.data_utils import load_dataset_EA, prepare_candidate_set_MetFrag
from msmsrt_scorer.lib.evaluation_tools import get_topk_performance_from_scores, evaluate_parameter_grid

# Disable warnings from RDKit
LG = RDLogger.logger()
LG.setLevel(RDLogger.CRITICAL)


def load_data(args, sample_idx):
    """
    Wrapper around the data loading function accessing the SQLite DB. It takes as input the script parameters,
    establishes the DB connection and loads returns the requested data.

    :param args: argparse.ArgumentParser() object, holding the program parameters
    :param sample_idx: integer, index of random data sub-sample to load

    :return: challenge and candidates, dictionaries
    """
    # The preference scores are not used for the MetFrag comparison (MetFrag uses LogP values as preference scores
    # and calculates those internally. However, due to the implementation of 'load_dataset_EA' we need to specify some
    # preference model.
    pref_model = {"training_dataset": "MEOH_AND_CASMI_JOINT", "keep_test_molecules": False,
                  "estimator": "ranksvm", "molecule_representation": "substructure_count"}

    with sqlite3.connect("file:" + args.database_fn + "?mode=ro", uri=True) as db:
        challenges, candidates = load_dataset_EA(
            db, participant=args.participant, prefmodel=pref_model, ion_mode=args.ion_mode,
            max_n_cand=args.max_n_cand, sort_candidates_by_ms2_score=args.sort_candidates_by_ms2_score,
            sample_idx=sample_idx)

    return challenges, candidates


def _load_data_from_metfrag_output(
        args, sample_idx, cnds_db,
        metfag_stg_fn="order_score=cdk__scores=FragmenterScore,RetentionTimeScore__score_weights=1.00000,1.00000"):
    """
    Load the csv-file that MetFrag outputs
    """
    # Get spectra identifier of the current split
    spec_ids = [v["name"] for v in cnds_db.values()]

    # Directory containing the challenge and sample specific scores
    chlg_score_dir = os.path.join(args.score_dir, "challenge=EA_%s__sample_idx=%03d" % (args.ion_mode, sample_idx))

    # Filename of the archive containing the candidate scores
    tar_fn = os.path.join(chlg_score_dir, metfag_stg_fn + ".tar.gz")

    cnds_csv = OrderedDict()
    with tarfile.open(tar_fn, "r:gz") as tar_file:
        for i, spec_id in enumerate(spec_ids):
            csv_file = tar_file.extractfile(os.path.join(metfag_stg_fn, spec_id + ".csv"))
            if csv_file is None:
                raise RuntimeError("Cannot open extract: '%s' for challenge=EA_%s and sample_id=%d"
                                   % (spec_id, args.ion_mode, sample_idx))

            # Read the scores
            scores_df = pd.read_csv(csv_file,
                                    usecols=["InChI", "InChIKey1", "Score", "FragmenterScore", "RetentionTimeScore"])
            scores_df = scores_df.sort_values("Score", ascending=False).drop_duplicates("InChIKey1", keep="first")

            if not args.sort_candidates_by_ms2_score:
                scores_df = scores_df.sample(frac=1, random_state=i)

            _structures = [MolToInchi(MolFromInchi(inchi), options="-SNon") for inchi in scores_df.InChI]
            assert (None not in _structures), "Could not convert all InChI to 2D-InChI"

            cnds_csv[i] = {"name": spec_id,
                           "structure": np.array(_structures),
                           "score": scores_df.FragmenterScore.values,
                           "pref_score": scores_df.RetentionTimeScore.values,
                           "n_cand": scores_df.shape[0]}

            # We ensure here, that the maximum score is 1.0 ==> s_i / s_max
            for _scr in ["score", "pref_score"]:
                if np.max(cnds_csv[i][_scr]) == 0.0:
                    cnds_csv[i][_scr] = np.ones_like(cnds_csv[i][_scr])
                else:
                    cnds_csv[i][_scr] /= np.max(cnds_csv[i][_scr])

    # Add information about the correct candidate
    for i in cnds_csv:
        try:
            _crct_idx = cnds_csv[i]["structure"].tolist().index(cnds_db[i]["correct_structure"])
        except ValueError:
            raise ValueError("Cannot find correct candidate: challenge=EA_%s, sample_idx=%03d"
                             % (args.ion_mode, sample_idx))

        cnds_csv[i]["index_of_correct_structure"] = _crct_idx

        assert (cnds_csv[i]["name"] == cnds_db[i]["name"])

        if cnds_csv[i]["n_cand"] != cnds_db[i]["n_cand"]:
            print("%s, n_cand_db=%d, n_cand_csv=%d"
                  % (cnds_csv[i]["name"], cnds_db[i]["n_cand"], cnds_csv[i]["n_cand"]))

    return cnds_csv


def _run_for_grid_item(cnds_csv, D):
    """
    Calculate the marginals for all ms-features in 'cnds_csv'. Here, we simply combine the

    :param cnds_csv: candidate dictionary, loaded from the CSV-files
    :param D: scalar, retention order weight
    :return: tuple (
        (D, None): hyper parameters that where used, i.e. retention order weight, no sigmoid parameter neded
        rep: 0
        marg: OrderedDict, containing the marginals (values) for all ms-features i (keys)
        Z_max: None
        p_max: -1
    )
    """
    # Extract marginals, which are just the (combined) MetFrag-scores weighted by D
    #   (1 - D) * FragmenterScore + D * RetentionTimeScore
    marg = {i: ((1 - D) * cnd["score"] + D * cnd["pref_score"]) for i, cnd in cnds_csv.items()}

    return (D, None), 0, marg, None, -1.0


def run_parameter_grid(args, sub_set, h_param_grid, sample_idx):
    """
    Get marginals for the different hyper parameter grid values of D for the 'sub_set' if ms-features.

    :param args: argparse.ArgumentParser() object, holding the program parameters
    :param sub_set: list, indices that should be used for the grid-search, e.g. the training set
    :param h_param_grid: dictionary of lists, D value grid. The output format is defined by the scikit-learn
        function 'model_selection.ParameterGrid'.
    :param sample_idx: integer, index of random data sub-sample to load

    :return: tuple (
        res: results for each D value grid element
        candidates: candidates loaded from the CSV
        h_param_grid: pass through
        1: number of "trees", just for compatibility here
    )
    """
    # Load data
    challenges, candidates = load_data(args, sample_idx)

    # Prepare and subset candidate sets
    cnds_db = prepare_candidate_set_MetFrag(challenges, candidates, sub_set, ms2_idc=range(len(sub_set)))

    # Load scores directly from MetFrag output
    cnds_csv = _load_data_from_metfrag_output(args, sample_idx, cnds_db)

    # Run score combination for the D-grid
    res = [_run_for_grid_item(cnds_csv, params["D"]) for params in h_param_grid]

    return res, cnds_csv, h_param_grid, 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--param_selection_measure", type=str, default="topk_auc",
                            choices=["topk_auc", "ndcg", "p_marg", "p_max", "un_topk_auc", "un_p_marg"],
                            help="Criteria for the selection of the best (D, k)-tuple (hyper-parameters, see Section "
                                 "3.4 and 4.2.2). In the paper 'topk_auc' (top20AUC) was used.")

    arg_parser.add_argument("--D_value_grid", nargs="+", type=float,
                            help="Grid-values for the retention order weight. (1 - D) * llh(MS) + D * llh(RT)")

    # Optional parameters dataset
    arg_parser.add_argument("--participant", type=str, default="MetFrag_2.4.5__8afe4a14",
                            choices=["MetFrag_2.4.5__8afe4a14"], help="MS2 scoring approach to be used.")

    arg_parser.add_argument("--max_n_cand", type=float, default=np.inf,
                            help="Limit the maximum number of candidates per ms-feature. We use all candidates in the"
                                 "paper. This option is mainly used for development.")

    arg_parser.add_argument("--sort_candidates_by_ms2_score", action="store_true",
                            help="Should the molecular candidates for each ms-feature be sorted in descending order "
                                 "(largest MS2 scores first) while loading the data. This option was mainly added to "
                                 "remove leaked information in the case where we simulate missing MS2 information. The "
                                 "ordering would be that leaked information. For the paper we always load the "
                                 "candidates in a random order.")

    # Optional parameters data
    arg_parser.add_argument("--database_fn", type=str, help="Path to the score SQLite DB.")

    arg_parser.add_argument("--score_dir", type=str,
                            default="/home/bach/Documents/doctoral/projects/local_casmi_db/data/EA/scores_triton/",
                            help="Path to the MetFrag scores (that have not been added yet to the DB). Those are used " 
                                 "to extract the preference scores (relationship between predicted LogP and observed "
                                 "retention time).")

    arg_parser.add_argument("--mode", type=str, default="debug", choices=["debug", "application"])

    arg_parser.add_argument("--base_odir", type=str, default="results__subsetFix__MetFrag",
                            help="Base directory to store the results and output files.")

    # Optional parameters evaluation
    arg_parser.add_argument("--ion_mode", type=str, default="positive",
                            help="Load spectra only of the specified ionization mode. In the paper we always analyzed "
                                 "negative and positive mode spectra separately.")

    arg_parser.add_argument("--max_n_ms2", type=int, default=100,
                            help="Number of MS2 spectra used for the evaluation of the score integration framework. "
                                 "For each sub-sample (see '--n_samples') we use this parameter to define the test set "
                                 "size. The remaining spectra are used to determine the best (D, k)-tuple. See Section "
                                 "3.1 for details on the training and test split sizes.")

    arg_parser.add_argument("--n_samples", type=int, default=100,
                            help="Number of random dataset sub-samples. Those have been already pre-sampled and are "
                                 "stored in the DB (see Section 3.1)")

    args = arg_parser.parse_args()
    print(str(args))

    # Define output directory
    odir = os.path.join(args.base_odir, args.mode,
                        "__".join(["param_selection_measure=%s" % args.param_selection_measure]),
                        "__".join(["ion_mode=%s" % args.ion_mode,
                                   "participant=%s" % args.participant,
                                   "max_n_cand=%.0f" % args.max_n_cand,
                                   "pref_model=%s" % "cdk",
                                   "sort_candidates_by_ms2_score=%d" % args.sort_candidates_by_ms2_score]))
    os.makedirs(odir, exist_ok=True)
    os.makedirs(os.path.join(odir, "candidates"), exist_ok=True)

    # Set up hyper-parameter grid
    h_param_grid = ParameterGrid({"D": args.D_value_grid, "k": [None]})
    print("Number of grid-pairs: %d" % len(h_param_grid))

    measure_df = pd.DataFrame(
        columns=["set", "D", "k", "p_marg", "p_max", "topk_auc", "top1", "top3", "top5", "top10", "top20", "ndcg"])
    opt_param_df = pd.DataFrame(columns=["sample", "D"])

    for s in range(args.n_samples):
        # Load data: This is only used to verify we are handling the CSV-output files from MetFrag correctly
        challenges, candidates = load_data(args, sample_idx=s)

        print("rep=%d/%d" % (s + 1, args.n_samples))

        train_set, test_set = next(ShuffleSplit(test_size=args.max_n_ms2, random_state=s).split(candidates))
        print("\tn_train=%d, n_test=%d" % (len(train_set), len(test_set)))

        # Perform grid-search on training set
        df_train = evaluate_parameter_grid(*run_parameter_grid(args, train_set, h_param_grid, s))
        df_train["set"] = "train"
        df_train["sample"] = s
        measure_df = pd.concat([measure_df, df_train], sort=True, axis=0)

        # Get top-k baseline performance of the test set
        # Prepare the candidate set, i.e. sub-setting, sampling the MS2 (if needed), ...
        cnds_test = prepare_candidate_set_MetFrag(challenges, candidates, test_set, ms2_idc=range(len(test_set)))

        print("\tTop-5 parameters:")
        print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
              .nlargest(5, columns=args.param_selection_measure))
        print("\tFlop-5 parameters:")
        print(df_train[["D", "k", args.param_selection_measure, "top1", "top5"]]
              .nsmallest(5, columns=args.param_selection_measure))

        # Get the optimal (D, k) parameter tuple
        _idxmax = df_train[args.param_selection_measure].idxmax()
        D_opt = df_train.loc[_idxmax, "D"]
        print("\tOptimal parameters: D=%f, k=None" % D_opt)
        opt_param_df.loc[s] = [s, D_opt]

        # Run Forward-Backward algorithm on the test set
        cnds_test_csv = _load_data_from_metfrag_output(args, s, cnds_test)
        res = _run_for_grid_item(cnds_test_csv, D_opt)

        # Average the marginals
        marg_test = {i: res[2][i] for i in cnds_test}

        # Calculate the top-k performance on the test set using RT+MS
        topk_test_casmi = {"ms_rt": get_topk_performance_from_scores(cnds_test_csv, marg_test, method="casmi2016"),
                           "baseline": get_topk_performance_from_scores(cnds_test_csv, None, method="casmi2016"),
                           "baseline_db": get_topk_performance_from_scores(cnds_test, None, method="casmi2016")}
        print("\tMS only (CSV): top1=%d (%.2f%%), top5=%d (%.2f%%), top10=%d (%.2f%%), top20=%d (%.2f%%)\n"
              "\tMS only (DB): top1=%d (%.2f%%), top5=%d (%.2f%%), top10=%d (%.2f%%), top20=%d (%.2f%%)\n"
              "\tMS + RT: top1=%d (%.2f%%), top5=%d (%.2f%%), top10=%d (%.2f%%), top20=%d (%.2f%%)\n"
              % (topk_test_casmi["baseline"][0][0], topk_test_casmi["baseline"][1][0],
                 topk_test_casmi["baseline"][0][4], topk_test_casmi["baseline"][1][4],
                 topk_test_casmi["baseline"][0][9], topk_test_casmi["baseline"][1][9],
                 topk_test_casmi["baseline"][0][19], topk_test_casmi["baseline"][1][19],
                 topk_test_casmi["baseline_db"][0][0], topk_test_casmi["baseline_db"][1][0],
                 topk_test_casmi["baseline_db"][0][4], topk_test_casmi["baseline_db"][1][4],
                 topk_test_casmi["baseline_db"][0][9], topk_test_casmi["baseline_db"][1][9],
                 topk_test_casmi["baseline_db"][0][19], topk_test_casmi["baseline_db"][1][19],
                 topk_test_casmi["ms_rt"][0][0], topk_test_casmi["ms_rt"][1][0],
                 topk_test_casmi["ms_rt"][0][4], topk_test_casmi["ms_rt"][1][4],
                 topk_test_casmi["ms_rt"][0][9], topk_test_casmi["ms_rt"][1][9],
                 topk_test_casmi["ms_rt"][0][19], topk_test_casmi["ms_rt"][1][19]))

        topk_test_csi = {"ms_rt": get_topk_performance_from_scores(cnds_test_csv, marg_test, method="csifingerid"),
                         "baseline": get_topk_performance_from_scores(cnds_test_csv, None, method="csifingerid")}

        with gzip.open(os.path.join(odir, "__".join(["topk_csi",
                                                     "max_n_ms2=%03d" % args.max_n_ms2,
                                                     "sample_id=%03d" % s]) + ".pkl.gz"), "wb+") as of:
            pickle.dump(topk_test_csi, of, protocol=pickle.HIGHEST_PROTOCOL)

        with gzip.open(os.path.join(odir, "__".join(["topk_casmi",
                                                     "max_n_ms2=%03d" % args.max_n_ms2,
                                                     "sample_id=%03d" % s]) + ".pkl.gz"), "wb+") as of:
            pickle.dump(topk_test_casmi, of, protocol=pickle.HIGHEST_PROTOCOL)

        # Write out information to reload candidates
        ofn = os.path.join(odir, "candidates", "__".join(["candidates",
                                                          "max_n_ms2=%03d" % args.max_n_ms2,
                                                          "sample_id=%03d" % s]) + ".pkl.gz")
        if not os.path.exists(ofn):
            with gzip.open(ofn, "wb+") as of:
                pickle.dump({"test_set": test_set, "random_state": s, "max_n_ms2": args.max_n_ms2}, of,
                            protocol=pickle.HIGHEST_PROTOCOL)

    # Write out stuff for the ranking measurements and parameter selection evaluation
    measure_df.to_csv(os.path.join(odir, "measures.csv"), index=False)

    if args.mode in ["application", "debug"]:
        opt_param_df.to_csv(os.path.join(odir, "opt_params.csv"), index=False)

    sys.exit(0)
