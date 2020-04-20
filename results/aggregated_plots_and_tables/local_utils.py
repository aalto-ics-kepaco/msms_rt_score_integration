import os
import numpy as np

def IDIR_CASMI(
    base_dir="results", mode="development", make_order_prob="sigmoid",
    D_value_method="fixed", ion_mode="positive", participant="MetFrag_2.4.5__8afe4a14", pref_model="c6d6f521",
    max_n_cand=np.inf, sort_candidates_by_ms2_score=False, tree_method="random", n_random_trees=16,
    min_min_rt_diff=0.0, max_min_rt_diff=0.0, param_selection_measure=None, norm_order_scores=None,
    use_global_parameter_selection=None, margin_type=None, restrict_candidates_to_correct_mf=None):
    
    # Define output directory
    if tree_method == "random":
        _tree_description = "random__n_trees=%d" % n_random_trees
    elif tree_method == "retention_time":
        _tree_description = "retention_time__min=%.2f_max=%.2f" % (min_min_rt_diff, max_min_rt_diff)
    elif tree_method == "chain":
        _tree_description = tree_method
    elif tree_method is None:
        pass
    else:
        raise ValueError("Invalid tree method: '%s'" % tree_method)

    # Get first level directory name
    _tmp = []
    if tree_method is not None:
        _tmp.append("tree_method=%s" % _tree_description)
    if make_order_prob is not None:
        _tmp.append("make_order_prob=%s" % make_order_prob)
    if D_value_method is not None:
        _tmp.append("D_value_method=%s" % D_value_method)
    if param_selection_measure is not None:
        _tmp.append("param_selection_measure=%s" % param_selection_measure)
    if use_global_parameter_selection is not None:
        _tmp.append("globparam=%d" % use_global_parameter_selection)
    if norm_order_scores is not None:
        _tmp.append("norm_order_scores=%d" % norm_order_scores)
    if margin_type is not None:
        _tmp.append("mtype=%s" % margin_type)
    if restrict_candidates_to_correct_mf is not None:
        _tmp.append("crcmf=%d" % restrict_candidates_to_correct_mf)
        
    idir = os.path.join(
        base_dir, mode, "__".join(_tmp),
        "__".join(["ion_mode=%s" % ion_mode,
                   "participant=%s" % participant,
                   "max_n_cand=%.0f" % max_n_cand,
                   "pref_model=%s" % pref_model,
                   "sort_candidates_by_ms2_score=%d" % sort_candidates_by_ms2_score]))
    
    return idir


def IDIR_EA(
    base_dir="results", mode="development", make_order_prob="sigmoid",
    D_value_method="fixed", ion_mode="positive", participant="MetFrag_2.4.5__8afe4a14",
    max_n_cand=np.inf, sort_candidates_by_ms2_score=False, tree_method="random", n_random_trees=16,
    min_min_rt_diff=0.0, max_min_rt_diff=0.0, param_selection_measure=None, norm_order_scores=None,
    training_dataset="MEOH_AND_CASMI_JOINT", keep_test_molecules=False, estimator="ranksvm", 
    molecular_representation="substructure_count", use_global_parameter_selection=None, margin_type=None,
    norm_scores=None):
    
    # Define output directory
    if tree_method == "random":
        _tree_description = "random__n_trees=%d" % n_random_trees
    elif tree_method == "retention_time":
        _tree_description = "retention_time__min=%.2f_max=%.2f" % (min_min_rt_diff, max_min_rt_diff)
    elif tree_method == "chain":
        _tree_description = tree_method
    elif tree_method is None:
        pass
    else:
        raise ValueError("Invalid tree method: '%s'" % tree_method)

    # Get first level directory name
    _tmp = []
    if tree_method is not None:
        _tmp.append("tree_method=%s" % _tree_description)
    if make_order_prob is not None:
        _tmp.append("make_order_prob=%s" % make_order_prob)
    if D_value_method is not None:
        _tmp.append("D_value_method=%s" % D_value_method)
    if param_selection_measure is not None:
        _tmp.append("param_selection_measure=%s" % param_selection_measure)
    if use_global_parameter_selection is not None:
        _tmp.append("globparam=%d" % use_global_parameter_selection)
    if norm_order_scores is not None:
        _tmp.append("norm_order_scores=%d" % norm_order_scores)
    if norm_scores is not None:
        _tmp.append("norm_scores=%s" % norm_scores)
    if margin_type is not None:
        _tmp.append("mtype=%s" % margin_type)
        
    idir = os.path.join(
        base_dir, mode, 
        "__".join(_tmp),
        "__".join(["ion_mode=%s" % ion_mode,
                   "participant=%s" % participant,
                   "max_n_cand=%.0f" % max_n_cand,
                   "sort_candidates_by_ms2_score=%d" % sort_candidates_by_ms2_score]),
        "__".join(["trainset=%s" % training_dataset,
                   "keep_test=%d" % keep_test_molecules,
                   "est=%s" % estimator,
                   "mol_rep=%s" % molecular_representation]))
    
    return idir