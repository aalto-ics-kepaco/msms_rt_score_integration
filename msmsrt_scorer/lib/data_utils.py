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

import numpy as np
import sqlite3

from collections import OrderedDict
from copy import deepcopy
from scipy import stats
from sklearn.utils.random import check_random_state

from msmsrt_scorer.lib.cindex_measure import cindex


def minmax_kernel(X, Y=None):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    if Y is None:
        Y = X

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features for set A and B must match: %d vs %d." % (
            X.shape[1], Y.shape[1]))

    n_A, n_B = X.shape[0], Y.shape[0]

    min_K = np.zeros((n_A, n_B))
    max_K = np.zeros((n_A, n_B))

    # Dense Matrix Implementation
    for s in range(X.shape[1]):  # loop if the feature dimensions
        c_s_A = X[:, s].reshape(-1, 1)
        c_s_B = Y[:, s].reshape(-1, 1)

        # Check for empty features dimension
        if np.all(c_s_A == 0) and np.all(c_s_B == 0):
            continue

        min_K += np.minimum(c_s_A, c_s_B.T)
        max_K += np.maximum(c_s_A, c_s_B.T)

    K_mm = min_K / max_K

    return K_mm


def tanimoto_kernel(X, Y=None):
    """
    Tanimoto kernel function

    :param X: array-like, shape=(n_samples_A, n_features), binary feature matrix of set A
    :param Y: array-like, shape=(n_samples_B, n_features), binary feature matrix of set B
        or None, than Y = X

    :return array-like, shape=(n_samples_A, n_samples_B), tanimoto kernel matrix
    """
    if Y is None:
        Y = X

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features for set A and B must match: %d vs %d." % (
            X.shape[1], Y.shape[1]))

    XY = X @ Y.T
    XX = X.sum(axis=1).reshape(-1, 1)
    YY = Y.sum(axis=1).reshape(-1, 1)

    K_tan = XY / (XX + YY.T - XY)

    return K_tan


def is_sorted(li, ascending=True):
    """
    Check whether the given list is sorted.

    source: https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not

    :param li: input list
    :param ascending: boolean, indicating whether the list should be sorted ascending or not.
    :return: is sorted
    """
    if ascending:
        return all(li[i] <= li[i + 1] for i in range(len(li) - 1))
    else:
        return all(li[i] >= li[i + 1] for i in range(len(li) - 1))


def _in_sql(li):
    """
    Concatenates a list of strings to a SQLite ready string that can be used in combination
    with the 'in' statement.

    E.g.:
        ["house", "boat", "donkey"] --> "('house', 'boat', 'donkey')"


    :param li: list of strings
    :return: SQLite ready string for 'in' statement
    """
    return "(" + ",".join(["'%s'" % li for li in np.atleast_1d(li)]) + ")"


def sigmoid(x, k=1., x_0=0., T=1.):
    """
    Sigmoid function defined as in [1]:

        f(x) = T / (1 + exp(-k * (x - x_0)))

    Paper section 2.2.3

    :param x: scalar or array-like with shape=(n, m), values to transform using the logistic function.
    :param k: scalar, The logistic growth rate or steepness of the curve.
    :param x_0: scalar, The x-value of the sigmoid's midpoint
    :param T: scalar, The curve's maximum value
    :return: scalar or array-like with shape=(n, m), transformed input values

    Reference:
        [1] https://en.wikipedia.org/wiki/Logistic_function
    """
    return T / (1 + np.exp(-k * (x - x_0)))


def hinge_sigmoid(x, k=1., x_0=0):
    """
    Half-sided sigmoid function similar to the hinge-loss. For values below 0 (the sigmoid's midpoint) the
    shape of the function is sigmoid. On the other hand, for values above 0 the function is always 1.

    Paper section 2.2.3

    :param x: scalar or array-like with shape=(n, m), values to transform using the logistic function.
    :param k: scalar, The logistic growth rate or steepness of the curve.
    :param x_0: scalar, threshold at which the hinge-sigmoid becomes constantly 1
    :return: scalar or array-like with shape=(n, m), transformed input values
    """
    return np.minimum(sigmoid(x, k=k, T=2, x_0=x_0), 1)


def step_fun(x, eps=1e-10):
    """
    Step function

    Paper section 2.2.3

    :param x: scalar or array-like with shape=(n, m), values to transform using the logistic function.
    :param eps: scalar, value assigned to input values < 0
    :return: scalar or array-like with shape=(n, m), transformed input values
    """
    _x = np.ones_like(x)
    _x[x < 0] = eps
    return _x


def get_measured_mass(precursor_mz, adduct):
    """
    Function returning the mass of the precursor ion given its single hydrogen charged precursor mass per charge
    value (m/z)

    :param precursor_mz: scalar or array-like, precursor mass per charge value(s)
    :param adduct: string, Adduct that was used. Currently only [M+H] and [M-H] supported
    :return: scalar or array-like, precursor ion mass
    """
    if adduct == "[M+H]":
        measured_mass = precursor_mz - 1.007825
    elif adduct == "[M-H]":
        measured_mass = precursor_mz + 1.007825
    else:
        raise NotImplementedError("Only '[M+H]' and '[M-H]' adduct supported.")

    return measured_mass


def carratore(measured_mass, candidate_mass, ppm=5):
    """
    Calculates a score expressing the how well the monoisotopic mass of a molecular candidates
    fits to the measure precursor mass of an MS-feature. Formula taken from [1].

    Paper section 3.5.2

    :param measured_mass: scalar, mass of the precursor ion
    :param candidate_mass: scalar or array-like, monoisotopic masses of the candidate(s)
    :param ppm: scalar, accuracy of the mass spectrometry device

    :return: scalar or array-like, compatibility value of each candidate mass with the observed precursor ion mass

    Reference:
        [1] Del Carratore et al. 2018, Integrated Probabilistic Annotation (IPA): A Bayesian-based annotation
        method for metabolomic profiles integrating biochemical connections, isotope patterns and adduct relationships
    """
    assert np.isscalar(measured_mass)

    sigma = (ppm * measured_mass) / (2 * 10e6)
    return stats.norm.pdf(measured_mass - candidate_mass, loc=0, scale=sigma)


def _get_spec_name_pattern(dataset):
    """
    Returns the SQLite ready pattern, that can selected the spectra (by their name) of the specified dataset.

    :param dataset: string, dataset to load (either 'EA' or 'CASMI')
    :return: string
    """
    if dataset == "EA":
        spec_name_pattern = "EAX%"
    elif dataset == "CASMI":
        spec_name_pattern = "Challenge-%"
    else:
        raise ValueError("Invalid dataset '%s'" % dataset)

    return spec_name_pattern


def _load_challenges(db, mode, dataset):
    """
    Load (spectrum name, rt, number of candidates) tuples for the given (dataset, ionization mode) combination.

    In the paper we have: (CASMI, negative), (CASMI, positive), (EA, negative) and (EA, positive)

    See section 3.1

    :param db: sqlite3 connection, to the local database
    :param mode: string, ionization mode (either 'positive' or 'negative')
    :param dataset: string, dataset to load (either 'EA' or 'CASMI')
    :return: sqlite3 cursor, to load the results from the database
    """
    res = db.execute(
        "SELECT spectrum, s.rt, count(candidate) FROM candidates_spectra"
        "    INNER JOIN spectra s on candidates_spectra.spectrum = s.name"  
        "    WHERE ionization_mode IS ? AND name LIKE ?"
        "    GROUP BY spectrum, rt"
        "    ORDER BY rt", (mode, _get_spec_name_pattern(dataset)))

    return res


def _load_pref_scores(db, prefmodel):
    """
    Load all RankSVM preference scores from the database

    See section 2.2.3 and 3.2

    :param db: sqlite3 connection, to the local database
    :param prefmodel: string, hash identifying the desired preference model
    :return: sqlite3 cursor, to load the results from the database
    """
    return db.execute(
        "SELECT molecule, score FROM preference_scores_data"
        "    WHERE setting IS ?", (prefmodel, ))


def _load_cand_scores(db, challenge_name, participant, sort_candidates_by_ms2_score=True, random_state=None,
                      max_n_cand=np.inf, molecular_formula=None):
    """
    Loads the molecular candidate structures and their MS2 scores as described in [1]

    See section: 3.1

    :param db: sqlite3 connection, to the local database
    :param challenge_name: string, spectra identifier for which the candidate MS2 scores should be loaded
        (e.g. Challenge-034)
    :param participant: string, MS2 scorer method
    :param sort_candidates_by_ms2_score: boolean, indicating whether the candidates should be sorted (descendent) by
        their MS2 score
    :param random_state: None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    :param max_n_cand: scalar, maximum number of candidates to load. If np.inf call candidates are loaded.
    :param molecular_formula: string or None, restrict candidates to load to the specified molecular formula.
        If None all candidates are loaded

    :return: list of tuples, results from the database

    Note:
        https://stackoverflow.com/questions/1591909/group-by-behavior-when-no-aggregate-functions-are-present-in-the-select-clause

    Reference:
        [1] E. Schymanski et al. 2017, Critical Assessment of Small Molecule Identification 2016: automated methods
    """

    if molecular_formula is None:
        res = db.execute(
            "SELECT candidate, inchi2D, MAX(score) AS max_score, monoisotopic_mass "
            "   FROM spectra_candidate_scores"
            "   INNER JOIN molecules m ON spectra_candidate_scores.candidate = m.inchi"
            "   WHERE spectrum IS ? AND participant IS ?"
            "   GROUP BY inchikey1"
            "   ORDER BY max_score DESC", (challenge_name, participant)).fetchall()
    else:
        assert isinstance(molecular_formula, str)
        print("WARNING: Restrict candidate set to correct molecular structure only!")
        res = db.execute(
            "SELECT candidate, inchi2D, MAX(score) AS max_score, monoisotopic_mass "
            "   FROM spectra_candidate_scores"
            "   INNER JOIN molecules m ON spectra_candidate_scores.candidate = m.inchi"
            "   WHERE spectrum IS ? AND participant IS ? AND m.molecular_formula IS ?"
            "   GROUP BY inchikey1"
            "   ORDER BY max_score DESC", (challenge_name, participant, molecular_formula)).fetchall()

    # Note: The candidates are sorted (descendant) by their MS2 scores

    # Take the top n candidates, if desired
    if max_n_cand < np.inf:
        res = res[:max_n_cand]

    # Shuffle the candidates, if desired
    # Note: This is useful, if one wants to remove all MS2 information from the candidate list.
    if not sort_candidates_by_ms2_score:
        check_random_state(random_state).shuffle(res)  # in-place

    return res


def _load_correct_molecular_structures(db, mode, dataset):
    """
    Returns the information regarding the correct molecular structure for each spectra in the specified
    (dataset, ionization) tuple.

    :param db: sqlite3 connection, to the local database
    :param mode: string, ionization mode (either 'positive' or 'negative')
    :param dataset: string, dataset to load (either 'EA' or 'CASMI')
    :return: sqlite3 cursor, to load the results from the database
    """
    res = db.execute(
        "SELECT molecule, inchi2D, rt, precursor_mz, adduct, molecular_formula, name FROM spectra"
        "    INNER JOIN molecules m ON spectra.molecule = m.inchi"
        "    WHERE ionization_mode IS ? AND name LIKE ?"
        "    ORDER BY rt", (mode, _get_spec_name_pattern(dataset)))

    return res


def _load_fps(db, inchis, fps_name="morgan_binary"):
    """
    Returns the fingerprints and an appropriate kernel function for the provided molecular structures. The
    fingerprints are loaded from the local database and the molecules are identifier by their InChI.

    :param db: sqlite3 connection, to the local database
    :param inchis: list of strings, InChIs of the molecules for which the fingerprints should be loaded
    :param fps_name: string, name of the desired fingerprint definition (must be present in the database)
    :return: tuple:
        array-like, shape=(n_mol, n_bits), fingerprint matrix
        callable, function that can be used to calculate the kernel similarity values using the fingerprints
    """
    inchis = np.atleast_1d(inchis)

    # Get dimension of the fingerprint
    nbits = db.execute("SELECT length FROM fingerprints_meta WHERE name IS ?", (fps_name, )).fetchall()[0][0]

    # Create output fingerprint matrix
    fps = np.zeros((len(inchis), nbits), dtype=int)

    if fps_name in ["morgan_binary", "morgan_binary_feat"]:
        kernel_fun = tanimoto_kernel
    elif fps_name in ["substructure_count"]:
        kernel_fun = minmax_kernel
    else:
        raise ValueError("Invalid fingerprints: '%s'" % fps_name)

    # Read the fingerprints and fill the fingerprint matrix
    for idx, inchi in enumerate(inchis):
        res = db.execute("SELECT %s FROM fingerprints_data WHERE molecule IS ?" % fps_name, (inchi, )).fetchall()
        assert (len(res) > 0)

        if fps_name in ["morgan_binary", "morgan_binary_feat"]:
            fps[idx, list(map(int, res[0][0].split(",")))] = 1
        elif fps_name in ["substructure_count"]:
            _data = eval("{" + res[0][0] + "}")
            fps[idx, list(_data.keys())] = list(_data.values())

    return fps, kernel_fun


def load_dataset_EA(db, ion_mode, participant, prefmodel, sample_idx, max_n_cand=np.inf,
                    sort_candidates_by_ms2_score=False, add_similarity_with_correct_structure=False, verbose=False,
                    fps_for_similarity_calculation="morgan_binary_feat"):
    """
    Load EA-dataset from local database.

    :param db: sqlite3 connection, to the local database
    :param ion_mode: string, ionization mode for which data data should be loaded. Can be 'positive' or 'negative'.
    :param participant: string, identifier of the method ("participant" in the DB) used to calculated the MS2 scores
        for the candidates
    :param prefmodel: dictionary, containing the specifications of the preference model for which the candidate
        preference scores should be loaded from the DB

        Example: {"training_dataset": "MEOH_AND_CASMI_JOINT", "keep_test_molecules": False, "estimator": "ranksvm",
                  "molecule_representation": "substructure_count"}
    :param sample_idx: integer, index of the random subsample to be loaded for evaluation
    :param max_n_cand: integer, maximum number of candidates per spectrum (default=inf)
    :param sort_candidates_by_ms2_score: boolean, indicating whether the candidates should be ordered by their MS2 score
        (default=False)
    :param add_similarity_with_correct_structure: boolean, indicating whether the similarity of each candidate with the
        correct molecular structure should be calculated
    :param fps_for_similarity_calculation: string, name of the fingerprint definition used to calculate the candidate
        similarity
    :param verbose: boolean indicating, whether debug output should be printed

    :return: tuple
        OrderedDict of dicts, containing the information for the challenge spectra.
            Spectra are sorted with increasing RT. 'Key' is an integer indexing the spectra and 'value' is a dictionary
            containing the spectra information:

                {"name": Identifier of the spectrum,
                 "retention_time": Retention time at which the spectrum has been measured,
                 "n_cand": Number of molecular candidates,
                 "correct_structure_3D": InChI (with all information) of the spectrum's (correct) molecular structure,
                 "correct_structure": InChI (without stereo-information) of the spectrum's (correct) molecular structure,
                 "precursor_mz": Mass-per-charge of the precursor corresponding to the spectrum,
                 "adduct": Adduct (typically depending on the ionization) that produced the precursor ion,
                 "correct_molecular_formula": Molecular formula of the spectrum's (correct) molecular structure}

        OrderedDict of dicts, containing the information of the molecular candidates corresponding to each spectrum
            The index ('key') of the candidate sets are the same as for the challenge spectra. The 'value' are the
            dictionaries containing the candidate information:

                {"structure": List of InChI (without stereo-information) of the molecular candidates,
                 "structure_3D": List of InChI (with all information) of the molecular candidates,
                 "score": MS2 score for each candidate r for the given spectrum i, theta_ir,
                 "pref_score": RankSVM preference scores for each candidate r of spectrum i: w^T phi_ir,
                 "structure_mass": Mono-isotopic mass of each candidate structure,
                 "n_cand": Number of molecular candidate structures,
                 "unknown_mass": Mass of the 'unknown' compound 'behind' the spectrum i
                 "index_of_correct_structure": Index of the correct candidate in the list of structures (determined by
                    its InChI without stereo-information),
                 "similarity_with_correct_structure": Similarity of each candidate with the correct candidate,
                    calculated using fingerprints}
    """
    # Get challenges / spectra belonging to the current 'sample_idx'
    challenges_in_random_sample = db.execute("SELECT spectrum FROM challenges_spectra_sample"
                                             "    WHERE challenge IS ? AND sample_idx IS ?",
                                             ("EA_" + ion_mode, sample_idx)).fetchall()
    challenges_in_random_sample = list(zip(*challenges_in_random_sample))[0]

    # Load challenges
    challenges = OrderedDict()
    i = 0
    for r in _load_challenges(db, mode=ion_mode, dataset="EA"):
        if r[0] in challenges_in_random_sample:
            assert(r[1] > 0), "All challenges should have at least one candidate."
            challenges[i] = {"name": r[0], "retention_time": r[1], "n_cand": r[2]}
            i += 1
    n_spec = len(challenges)
    if verbose:
        print("Number of spectra (ionization=%s, sample=%d): %d" % (ion_mode, sample_idx, n_spec))

    # Load correct molecular structures for the challenges
    _rt = - np.inf
    i = 0
    for r in _load_correct_molecular_structures(db, mode=ion_mode, dataset="EA"):
        if r[-1] in challenges_in_random_sample:
            assert (r[-1] == challenges[i]["name"])
            assert (r[2] == challenges[i]["retention_time"])

            challenges[i]["correct_structure_3D"] = r[0]  # full inchi
            challenges[i]["correct_structure"] = r[1]  # inchi-2D
            challenges[i]["precursor_mz"] = r[3]  # precursor-mz
            challenges[i]["adduct"] = r[4]  # adduct
            challenges[i]["correct_molecular_formula"] = r[5]  # molecular formula

            if _rt > r[2]:
                raise Exception("Retention times are not ordered: t_(i-1) = %.4f, t_i = %.4f" % (_rt, r[2]))
            elif _rt == r[2]:
                if verbose:
                    print("Retention times of consecutive MS2 are equal: %s = %.2f, %s = %.2f"
                          % (challenges[i - 1]["name"], _rt, challenges[i]["name"], r[2]))
            _rt = r[2]

            i += 1

    # Load preference scores for all molecules
    prefmodel_hash = db.execute("SELECT name FROM preference_scores_meta"
                                "    WHERE training_dataset IS ?"
                                "      AND keep_test_molecules IS ?"
                                "      AND estimator IS ?"
                                "      AND molecule_representation IS ?"
                                "      AND challenge IS ?"
                                "      AND sample_idx IS ?",
                                (prefmodel["training_dataset"], "True" if prefmodel["keep_test_molecules"] else "False",
                                 prefmodel["estimator"], prefmodel["molecule_representation"],
                                 "EA_" + ion_mode, sample_idx)).fetchall()
    assert (len(prefmodel_hash) == 1)
    pref_scores = {r[0]: r[1] for r in _load_pref_scores(db, prefmodel_hash[0][0])}

    # Load candidate scores
    candidates = OrderedDict()

    for i, chlg in challenges.items():
        res = _load_cand_scores(db, chlg["name"], participant=participant,
                                sort_candidates_by_ms2_score=sort_candidates_by_ms2_score,
                                random_state=i, max_n_cand=max_n_cand)
        candidates[i] = {"structure": np.empty((len(res),), dtype=object),
                         "structure_3D": np.empty((len(res),), dtype=object),
                         "score": np.empty((len(res),), dtype=float),
                         "pref_score": np.empty((len(res),), dtype=float),
                         "structure_mass": np.empty((len(res),), dtype=float)}

        for idx, (inchi, inchi2D, ms_score, monoisotopic_mass) in enumerate(res):
            candidates[i]["structure_3D"][idx] = inchi  # full inchi
            candidates[i]["structure"][idx] = inchi2D  # inchi-2d
            candidates[i]["score"][idx] = ms_score
            candidates[i]["pref_score"][idx] = pref_scores[inchi]
            candidates[i]["structure_mass"][idx] = monoisotopic_mass

        candidates[i]["n_cand"] = candidates[i]["structure"].shape[0]
        challenges[i]["n_cand"] = candidates[i]["n_cand"]

        # Calculate the monoisotopic mass for the unknown compound from the mass-spectrum
        candidates[i]["unknown_mass"] = get_measured_mass(challenges[i]["precursor_mz"], challenges[i]["adduct"])

        assert (len(candidates[i]["score"]) == len(candidates[i]["pref_score"]))
        assert (len(candidates[i]["score"]) == len(candidates[i]["structure"]))

        # Find the index of the correct molecular structure
        try:
            candidates[i]["index_of_correct_structure"] = candidates[i]["structure"].tolist().index(
                challenges[i]["correct_structure"])
        except ValueError:
            candidates[i]["index_of_correct_structure"] = np.nan

        # Load fingerprints for all candidates and calculate their similarity with the correct candidate
        if add_similarity_with_correct_structure:
            _fps_of_correct_structure, _kernel_fun = _load_fps(
                db, candidates[i]["structure_3D"][candidates[i]["index_of_correct_structure"]],
                fps_for_similarity_calculation)
            _fps_of_candidates, _kernel_fun = _load_fps(db, candidates[i]["structure_3D"],
                                                        fps_for_similarity_calculation)

            _sim = _kernel_fun(_fps_of_candidates, _fps_of_correct_structure, shallow_input_check=True).flatten()
            assert(_sim.shape == (candidates[i]["n_cand"],))
            assert(_sim[candidates[i]["index_of_correct_structure"]] == 1.0)

            candidates[i]["similarity_with_correct_structure"] = _sim

    _rts = [chlg["retention_time"]
            for chlg, cnd
            in zip(challenges.values(), candidates.values())
            if not np.isnan(cnd["index_of_correct_structure"])]
    _wtx = [cnd["pref_score"][cnd["index_of_correct_structure"]]
            for cnd
            in candidates.values()
            if not np.isnan(cnd["index_of_correct_structure"])]

    if verbose:
        print("Cindex (correct structure): %.3f" % cindex(_rts, _wtx))

    return challenges, candidates


def load_dataset_CASMI(db, ion_mode, participant, prefmodel, max_n_cand=np.inf, sort_candidates_by_ms2_score=True,
                       verbose=True, restrict_candidates_to_correct_mf=False):
    """
    Load EA-dataset from local database.

    :param db: sqlite3 connection, to the local database
    :param ion_mode: string, ionization mode for which data data should be loaded. Can be 'positive' or 'negative'.
    :param participant: string, identifier of the method ("participant" in the DB) used to calculated the MS2 scores
        for the candidates
    :param prefmodel: string, hash identifying the preference model used to provide the RankSVM preference values for
        each molecular candidate
    :param max_n_cand: integer, maximum number of candidates per spectrum (default=inf)
    :param sort_candidates_by_ms2_score: boolean, indicating whether the candidates should be ordered by their MS2 score
        (default=False)
    :param verbose: boolean indicating, whether debug output should be printed
    :param restrict_candidates_to_correct_mf: boolean, indicating whether only those candidate should be loaded, that
        have the same molecular formula as the correct molecular structure

    :return: tuple
        OrderedDict of dicts, containing the information for the challenge spectra.
            Spectra are sorted with increasing RT. 'Key' is an integer indexing the spectra and 'value' is a dictionary
            containing the spectra information:

                {"name": Identifier of the spectrum,
                 "retention_time": Retention time at which the spectrum has been measured,
                 "n_cand": Number of molecular candidates,
                 "correct_structure_3D": InChI (with all information) of the spectrum's (correct) molecular structure,
                 "correct_structure": InChI (without stereo-information) of the spectrum's (correct) molecular structure,
                 "precursor_mz": Mass-per-charge of the precursor corresponding to the spectrum,
                 "adduct": Adduct (typically depending on the ionization) that produced the precursor ion,
                 "correct_molecular_formula": Molecular formula of the spectrum's (correct) molecular structure}

        OrderedDict of dicts, containing the information of the molecular candidates corresponding to each spectrum
            The index ('key') of the candidate sets are the same as for the challenge spectra. The 'value' are the
            dictionaries containing the candidate information:

                {"structure": List of InChI (without stereo-information) of the molecular candidates,
                 "structure_3D": List of InChI (with all information) of the molecular candidates,
                 "score": MS2 score for each candidate r for the given spectrum i, theta_ir,
                 "pref_score": RankSVM preference scores for each candidate r of spectrum i: w^T phi_ir,
                 "structure_mass": Mono-isotopic mass of each candidate structure,
                 "n_cand": Number of molecular candidate structures,
                 "unknown_mass": Mass of the 'unknown' compound 'behind' the spectrum i
                 "index_of_correct_structure": Index of the correct candidate in the list of structures (determined by
                    its InChI without stereo-information)}
    """

    # Load challenges
    challenges = OrderedDict()
    for i, r in enumerate(_load_challenges(db, mode=ion_mode, dataset="CASMI")):
        challenges[i] = {"name": r[0], "retention_time": r[1], "n_cand": r[2]}
    n_spec = len(challenges)
    if verbose:
        print("Number of spectra:", n_spec)

    # Load correct molecular structures for the challenges
    _rt = - np.inf
    for i, r in enumerate(_load_correct_molecular_structures(db, mode=ion_mode, dataset="CASMI")):
        assert (r[-1] == challenges[i]["name"])
        assert (r[2] == challenges[i]["retention_time"])

        challenges[i]["correct_structure_3D"] = r[0]  # full inchi
        challenges[i]["correct_structure"] = r[1]  # inchi-2D
        challenges[i]["precursor_mz"] = r[3]  # precursor-mz
        challenges[i]["adduct"] = r[4]  # adduct
        challenges[i]["correct_molecular_formula"] = r[5]  # molecular formula

        if _rt > r[2]:
            raise Exception("Retention times are not ordered: t_(i-1) = %.4f, t_i = %.4f" % (_rt, r[2]))
        elif _rt == r[2]:
            if verbose:
                print("Retention times of consecutive MS2 are equal: t_(i-1) = %.4f, t_i = %.4f" % (_rt, r[2]))
        _rt = r[2]

    # Load preference scores for all molecules
    pref_scores = {r[0]: r[1] for r in _load_pref_scores(db, prefmodel)}

    # Load candidate scores
    candidates = OrderedDict()
    for i, chlg in challenges.items():
        _ml = chlg["correct_molecular_formula"] if restrict_candidates_to_correct_mf else None
        res = _load_cand_scores(db, chlg["name"], participant=participant,
                                sort_candidates_by_ms2_score=sort_candidates_by_ms2_score,
                                random_state=i, max_n_cand=max_n_cand,
                                molecular_formula=_ml)

        candidates[i] = {"structure": np.empty((len(res),), dtype=object),
                         "structure_3D": np.empty((len(res),), dtype=object),
                         "score": np.empty((len(res),), dtype=float),
                         "pref_score": np.empty((len(res),), dtype=float),
                         "structure_mass": np.empty((len(res),), dtype=float)}

        for idx, (inchi, inchi2D, ms_score, monoisotopic_mass) in enumerate(res):
            candidates[i]["structure_3D"][idx] = inchi  # full inchi
            candidates[i]["structure"][idx] = inchi2D  # inchi-2d
            candidates[i]["score"][idx] = ms_score
            candidates[i]["pref_score"][idx] = pref_scores[inchi]
            candidates[i]["structure_mass"][idx] = monoisotopic_mass

        candidates[i]["n_cand"] = candidates[i]["structure"].shape[0]
        challenges[i]["n_cand"] = candidates[i]["n_cand"]

        # Calculate the monoisotopic mass for the unknown compound from the mass-spectrum
        candidates[i]["unknown_mass"] = get_measured_mass(challenges[i]["precursor_mz"], challenges[i]["adduct"])

        assert (len(candidates[i]["score"]) == len(candidates[i]["pref_score"]))
        assert (len(candidates[i]["score"]) == len(candidates[i]["structure"]))

        # Find the index of the correct molecular structure
        try:
            candidates[i]["index_of_correct_structure"] = candidates[i]["structure"].tolist().index(
                challenges[i]["correct_structure"])
        except ValueError:
            candidates[i]["index_of_correct_structure"] = np.nan

    _rts = [chlg["retention_time"]
            for chlg, cnd
            in zip(challenges.values(), candidates.values())
            if not np.isnan(cnd["index_of_correct_structure"])]
    _wtx = [cnd["pref_score"][cnd["index_of_correct_structure"]]
            for cnd
            in candidates.values()
            if not np.isnan(cnd["index_of_correct_structure"])]

    if verbose:
        print("Cindex (correct structure): %.3f" % cindex(_rts, _wtx))

    return challenges, candidates


def prepare_candidate_set_MetFrag(challenges, candidates, sub_set, n_ms2=None, random_state=None, ms2_idc=None,
                                  verbose=False, normalize=False) -> OrderedDict:
    """
    Pre-processing of the candidate for MetFrag:

        - Only consider sub-set of the original set, given by 'sub_set', e.g. to split into training and test
        - Properly regularize the MS-scores
        - Convert the MS-scores provided by MetFrag into prior-probabilities

    See section 2.2.2

    :param challenges: OrderedDict of dicts, output of load_dataset_*
    :param candidates: OrderedDict of dicts, output of load_dataset_*
    :param sub_set: list of indices, subset of spectra that should be returned
    :param n_ms2: scalar or None, number of spectra that should have MS2 scores (rather than MS1). If None, all spectra
        are with MS2 score.
    :param normalize: boolean, indicating whether the MS-scores should be normalized to sum up to one
    :param random_state: None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    :param ms2_idc: list of indices or None, list of spectra for which the MS2 scores (rather than MS1) should be
        returned. If None, all spectra will have MS2 scores.

    :return: OrderedDict of dicts with same structure as 'candidates' returned by load_dataset_*, but restricted to
        the specified spectra sub-set. RT, Spec-identifier and correct_structure information are added.
    """
    # Need to sort the index subset to ensure monotonic retention times
    sub_set = np.sort(sub_set)

    if n_ms2 is not None:
        # Specified how many challenges should have MS2 information. Eventually overwrite provided 'ms2_idc' values
        ms2_idc = check_random_state(random_state).permutation(np.arange(len(sub_set)))[:n_ms2]

    if ms2_idc is None:
        # Use for all challenge the MS2 information
        ms2_idc = np.arange(len(sub_set))

    # Copy data into sub-set candidate dictionary
    candidates_out = OrderedDict()
    for i_new, i_old in enumerate(sub_set):
        # Add retention times to the candidates dictionary
        candidates_out[i_new] = {"retention_time": challenges[i_old]["retention_time"],
                                 "name": challenges[i_old]["name"],
                                 "correct_structure": challenges[i_old]["correct_structure"]}

        if i_new in ms2_idc:
            candidates_out[i_new]["score"] = deepcopy(candidates[i_old]["score"])
            candidates_out[i_new]["has_ms2"] = True
        else:
            candidates_out[i_new]["score"] = carratore(candidates[i_old]["unknown_mass"],
                                                       candidates[i_old]["structure_mass"])
            candidates_out[i_new]["score"] /= np.max(candidates_out[i_new]["score"])
            candidates_out[i_new]["has_ms2"] = False

        # Copy remaining information
        candidates_out[i_new]["n_cand"] = candidates[i_old]["n_cand"]
        candidates_out[i_new]["index_of_correct_structure"] = candidates[i_old]["index_of_correct_structure"]
        candidates_out[i_new]["structure"] = candidates[i_old]["structure"]
        candidates_out[i_new]["structure_3D"] = candidates[i_old]["structure_3D"]
        candidates_out[i_new]["pref_score"] = candidates[i_old]["pref_score"]

        if "similarity_with_correct_structure" in candidates[i_old]:
            candidates_out[i_new]["similarity_with_correct_structure"] = \
                candidates[i_old]["similarity_with_correct_structure"]

    # Determine the constant 'c' added as regularizer to the MetFrag scores to avoid zero probabilities
    # 1) Collect all MS-scores from the sub-set candidates sets
    scores = np.hstack([cnd["score"] for cnd in candidates_out.values()])
    # 2) The regularization constant is 10-times smaller than the overall minimum of scores larger zero.
    c = np.min(scores[scores > 0]) / 10
    if verbose:
        print("Regularization constant: %.8f" % c)

    # Regularize the MS-scores, normalise and logarithmise them
    for i in candidates_out:
        candidates_out[i]["score"] = np.maximum(c, candidates_out[i]["score"])
        assert (np.all(candidates_out[i]["score"] > 0))

        # Probabilities must sum to one
        if normalize:
            candidates_out[i]["score"] /= np.sum(candidates_out[i]["score"])

        # Calculate the log-probabilities
        candidates_out[i]["log_score"] = np.log(candidates_out[i]["score"])

    return candidates_out


def prepare_candidate_set_IOKR(challenges, candidates, sub_set, n_ms2=None, random_state=None, ms2_idc=None,
                               verbose=False, normalize=False) -> OrderedDict:
    """
    Pre-processing of the candidate score for IOKR:

        - Only consider sub-set of the original set, given by 'sub_set', e.g. to split into training and test
        - Properly regularize the MS-scores
        - Convert the MS-scores provided by MetFrag into prior-probabilities

    See section 2.2.2

    :param challenges: OrderedDict of dicts, output of load_dataset_*
    :param candidates: OrderedDict of dicts, output of load_dataset_*
    :param sub_set: list of indices, subset of spectra that should be returned
    :param n_ms2: scalar or None, number of spectra that should have MS2 scores (rather than MS1). If None, all spectra
        are with MS2 score.
    :param normalize: boolean, indicating whether the MS-scores should be normalized to sum up to one
    :param random_state: None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    :param ms2_idc: list of indices or None, list of spectra for which the MS2 scores (rather than MS1) should be
        returned. If None, all spectra will have MS2 scores.

    :return: OrderedDict of dicts with same structure as 'candidates' returned by load_dataset_*, but restricted to
        the specified spectra sub-set. RT, Spec-identifier and correct_structure information are added.
    """
    # Need to sort the index subset to ensure monotonic retention times
    sub_set = np.sort(sub_set)

    if n_ms2 is not None:
        # Specified how many challenges should have MS2 information. Eventually overwrite provided 'ms2_idc' values
        ms2_idc = check_random_state(random_state).permutation(np.arange(len(sub_set)))[:n_ms2]

    if ms2_idc is None:
        # Use for all challenge the MS2 information
        ms2_idc = np.arange(len(sub_set))

    # Copy data into sub-set candidate dictionary
    candidates_out = OrderedDict()
    for i_new, i_old in enumerate(sub_set):
        # Add retention times to the candidates dictionary
        candidates_out[i_new] = {"retention_time": challenges[i_old]["retention_time"],
                                 "name": challenges[i_old]["name"],
                                 "correct_structure": challenges[i_old]["correct_structure"]}

        if i_new in ms2_idc:
            candidates_out[i_new]["score"] = deepcopy(candidates[i_old]["score"])
            candidates_out[i_new]["has_ms2"] = True
        else:
            candidates_out[i_new]["score"] = carratore(candidates[i_old]["unknown_mass"],
                                                       candidates[i_old]["structure_mass"])
            candidates_out[i_new]["score"] /= np.max(candidates_out[i_new]["score"])
            candidates_out[i_new]["has_ms2"] = False

        # Copy remaining information
        candidates_out[i_new]["n_cand"] = candidates[i_old]["n_cand"]
        candidates_out[i_new]["index_of_correct_structure"] = candidates[i_old]["index_of_correct_structure"]
        candidates_out[i_new]["structure"] = candidates[i_old]["structure"]
        candidates_out[i_new]["structure_3D"] = candidates[i_old]["structure_3D"]
        candidates_out[i_new]["pref_score"] = candidates[i_old]["pref_score"]

        if "similarity_with_correct_structure" in candidates[i_old]:
            candidates_out[i_new]["similarity_with_correct_structure"] = \
                candidates[i_old]["similarity_with_correct_structure"]

    # Determine the constant 'c' added as regularizer to the IOKR scores to avoid zero probabilities
    # 1) Collect all MS-scores from the sub-set candidates sets
    scores = np.hstack([cnd["score"] for cnd in candidates_out.values()])
    if verbose:
        print("Minimum MS-score: %.8f" % np.min(scores))
    # 2) Calculate the constant to make all scores >= 0
    if np.any(scores < 0):
        c1 = np.abs(np.min(scores))
        scores += c1
    else:
        c1 = 0.0
    # 3) The regularization constant is 10-times smaller than the overall minimum of scores larger zero.
    c2 = np.min(scores[scores > 0]) / 10
    if verbose:
        print("Regularization constant: c1=%.8f, c2=%.8f" % (c1, c2))

    # Regularize the MS-scores, normalise and logarithmise them
    for i in candidates_out:
        candidates_out[i]["score"] = np.maximum(c2, candidates_out[i]["score"] + c1)
        candidates_out[i]["score"] /= np.max(candidates_out[i]["score"])
        assert (np.all(candidates_out[i]["score"] > 0))

        # Probabilities must sum to one
        if normalize:
            candidates_out[i]["score"] /= np.sum(candidates_out[i]["score"])

        # Calculate the log-probabilities
        candidates_out[i]["log_score"] = np.log(candidates_out[i]["score"])

    return candidates_out


if __name__ == "__main__":
    # Path to the CASMI score database
    DBFN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"
    DBURL = "file:" + DBFN + "?mode=ro"

    # Ionization mode to consider
    MODE = "positive"

    # Load MS2 scores of specified method
    PARTICIPANT = "MetFrag_2.4.5__8afe4a14"

    # Preference score model
    PREFMODEL = {"training_dataset": "MEOH_AND_CASMI_JOINT", "keep_test_molecules": False, "estimator": "ranksvm",
                 "molecule_representation": "substructure_count"}

    with sqlite3.connect(DBURL, uri=True) as db:
        print("EA:", MODE)
        challenges, candidates = load_dataset_EA(db, MODE, PARTICIPANT, PREFMODEL, sample_idx=99,
                                                 sort_candidates_by_ms2_score=False, verbose=True,
                                                 add_similarity_with_correct_structure=False)
        print(len(challenges))

        print("CASMI", MODE)
        challenges, candidates = load_dataset_CASMI(db, MODE, PARTICIPANT, "c6d6f521",
                                                    sort_candidates_by_ms2_score=False, verbose=True)
        print(len(challenges))
