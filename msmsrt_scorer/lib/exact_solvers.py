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
import networkx as nx
import itertools as it

from collections import OrderedDict
from scipy.special import logsumexp
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize

import logging
logging.basicConfig(format='%(name)s: %(message)s', level=logging.WARNING)
LOGGER = logging.getLogger("FactorGraph")


class FactorGraph(object):
    def __init__(self, candidates, fac_rt, make_order_probs, order_probs=None, use_log_space=True, D=0.5,
                 norm_order_scores=False):
        """
        Class modelling the factor graph of the pairwise Markov random field (MRF) G = (V, E) described in Section 2.2.
        A factor graph contains a node for each variable (here the MS features {1, ..., i, ..., N}) and factors they
        are involved in. Those factors are shown in the probability distribution given in Equation (1) (see Section
        2.2.1):

            p(z) ~ prod {i in V} f_ms(z_i) * prod {(i, j) in E} f_rt(z_i, z_j),

        where f_ms and f_rt are the factors and i in {1, ..., N} are the variables.

        For the implementation of the factor graph and the corresponding inference algorithms we followed:

            [1] MacKay, D. J. "Information theory, inference and learning algorithms", Cambridge university press (2003)
            [2] Bishop, C. "Pattern Recognition and Machine Learning", Springer New York (2006)

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param fac_rt: list of tuples, [(i, j), ...], specification of the candidate set pairs (correspond to f_rt)

        :param make_order_probs: callable, function implementing Sigma of the edge potential function (see Section
            2.2.3). The function should be vectorized, i.e. being able to handle an np.ndarray as input.

        :param order_probs: dict of dicts, storing all pairwise edge potential matrices of size Z_i x Z_j for all pairs
            (i, j) corresponding to the f_rt factors. The matrices store the edge potential values for each candidate
            structure pair of a pair (i, j) (see Section 2.2.3). This parameter can be used if the matrices are pre-
            computed. By default, this parameter is None and the 'make_order_probs' function is used to calcualte the
            matrices directly from the given preference values in 'candidates'

        :param use_log_space: boolean, indicating whether the inference should be done in the log-space. That means,
            the edge potential value are logarithmised. Max-product --> Max-sum algorithm. This implementation currently
            only supports the implementation in the log-space.

        :param D: scalar or None, weight of the retention order information (see Section 2.2.4). If None, both retention
            order and mass spectrum scores are having weight 1.0.

        :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one. (default = False)
        """
        self.candidates = candidates
        self.use_log_space = use_log_space
        self.make_order_probs = make_order_probs
        self.norm_order_scores = norm_order_scores

        if D is None:
            self.D_ms = 1.0
            self.D_rt = 1.0
        elif np.isscalar(D):
            if D < 0.0 or D > 1.0:
                raise ValueError("D must be from [0, 1].")
            self.D_ms = 1.0 - D
            self.D_rt = D
        else:
            raise ValueError("D must either be a scalar or None.")

        # Get variables and factors for the MS2 scores
        self.var = list(candidates.keys())  # i = 1, 2, 3, ..., N
        self.fac_ms = [(i, i) for i in self.var]  # (1, 1), (2, 2), ..., (N, N)
        LOGGER.debug("MS-factors: %s" % str(self.fac_ms))

        # Set up factors and order probabilities using retention time information
        self.fac_rt = fac_rt
        if order_probs is None:
            # Calculate the order probs given the candidates, the required rt-factors and the transformation function.
            self.order_probs = self._precalculate_order_probs(self.candidates, self.fac_rt, self.make_order_probs,
                                                              self.norm_order_scores)
        else:
            self.order_probs = order_probs

        LOGGER.debug("RT-factors: %s" % str(self.fac_rt))

        for i, j in self.fac_rt:
            if i not in self.order_probs or j not in self.order_probs[i]:
                raise Exception("'order_probs' is missing the pair (i, j) = (%d, %d)." % (i, j))

        # Variables related to the Sum-Product algorithm: Marginals
        self.R_sum = None
        self.Q_sum = None

        # Variables related to the Max-Product algorithm: Maximum a-Posteriori
        self.R_max = None
        self.Q_max = None
        self.Par_max = None
        self.acc_max = None

    def _log_likelihood(self, Z) -> float:
        """
        Calculate the log-likelihood for a given candidate assignment:

            log lh(z) = (1 - D) sum {i in V} log f_ms(z_i) + D sum {(i, j) in E} log f_rt(z_i, z_j)

        Note: Not to confuse with the log probability distribution log p(z) for relates to the log-likelihood as follows:

            log p(z) = log (lh(z) / Z) = log lh(z) - log Z,

            where Z is the partition function or normalization constant.

        :param Z: array-like, shape=(n_spec,), indices of selected candidates for all spectra
        :return: scalar, log-likelihood of the given candidate assignment
        """
        llh_ms = 0.0
        llh_rt = 0.0

        # MS-score
        for i, _ in self.fac_ms:
            llh_ms += self.candidates[i]["log_score"][Z[i]]

        # RT-score
        for i, j in self.fac_rt:
            llh_rt += self.order_probs[i][j]["log_score"][Z[i]][Z[j]]

        return self.D_ms * llh_ms + self.D_rt * llh_rt

    def likelihood(self, Z, log=False) -> float:
        """
        Calculate the likelihood for a given candidate assignment:

            lh(Z) = prod {i in V} f_ms(z_i)^(1 - D) * prod {(i, j) in E} f_rt(z_i, z_j)^D

        Note: Not to confuse with the probability distribution p(z) for relates to the likelihood as follows:

            p(z) = lh(z) / Z,

            where Z is the partition function or normalization constant.

        :param Z: array-like, shape=(n_spec,), indices of selected candidates for all spectra
        :param log: boolean, indicating whether to the log-likelihood or likelihood value should be returned.
        :return: scalar, (log-)likelihood of the given candidate assignment
        """
        if log:
            val = self._log_likelihood(Z)
        else:
            val = np.exp(self._log_likelihood(Z))

        return val

    def get_candidate_list_graph(self) -> nx.Graph:
        """
        Returns the Markov random field (MRF) graph for visualization.

        :return: networkx graph representing the MRF:
            Nodes: correspond to the variables
            Node-labels: retention time, number of candidates
            Edges: correspond to the pairwise candidate list comparisons
            Edge-weights:
                weight_all: fraction of molecular candidates pairs, those predicted retention order does not obey the
                            observed one (all candidates)
                weight_20: fraction of molecular candidates pairs, those predicted retention order does not obey the
                           observed one (top-20 candidates)
        """
        G = nx.Graph()

        # Add nodes
        for i in self.var:
            G.add_node(i, retention_time=self.candidates[i]["retention_time"], n_cand=self.candidates[i]["n_cand"])

        # Add edges
        for i, j in self.fac_rt:
            n_cand_i, n_cand_j = self.order_probs[i][j]["score"].shape

            fp = self.order_probs[i][j]["score"] < 0.5
            tie = self.order_probs[i][j]["score"] == 0.5

            # Get number of molecule-pairs which do not obey the observed retention order (all)
            n_fp = np.sum(fp)
            n_tie = np.sum(tie) / 2.

            # Get number of molecule-pairs which do not obey the observed retention order (top-20)
            # Note: This assumes that the candidates are ordered by their MS2 score in descendent order
            min_i = np.minimum(n_cand_i, 20)
            min_j = np.minimum(n_cand_j, 20)

            n_fp_20 = np.sum(fp[:min_i, :min_j])
            n_tie_20 = np.sum(tie[:min_i, :min_j]) / 2.

            G.add_edge(i, j,
                       weight_all=(n_fp + n_tie) / (n_cand_i * n_cand_j),
                       weight_20=(n_fp_20 + n_tie_20) / (min_i * min_j))

        return G

    @staticmethod
    def _precalculate_order_probs(candidates, fac_rt, make_order_probs, norm_scores=False):
        """
        Calculate the order probabilities from the preference scores given the 'make_order_probs' function.

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param fac_rt: list of tuples, [(i, j), ...], specification of the candidate set pairs (correspond to f_rt)

        :param make_order_probs: callable, function implementing Sigma of the edge potential function (see Section
            2.2.3). The function should be vectorized, i.e. being able to handle an np.ndarray as input.

        :param norm_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one. (default = False)

        :return: dict of dicts, storing all pairwise edge potential matrices of size Z_i x Z_j for all pairs
            (i, j) corresponding to the f_rt factors. The matrices store the edge potential values for each candidate
            structure pair of a pair (i, j) (see Section 2.2.3).
        """
        order_probs = OrderedDict()
        for i, j in fac_rt:
            pref_i, pref_j = candidates[i]["pref_score"][:, np.newaxis], candidates[j]["pref_score"][np.newaxis, :]

            if i not in order_probs:
                order_probs[i] = OrderedDict()

            t_i, t_j = candidates[i]["retention_time"], candidates[j]["retention_time"]
            order_probs[i][j] = {"score": make_order_probs(np.sign(t_i - t_j) * (pref_i - pref_j),
                                                           loc=np.sign(t_i - t_j) * (t_i - t_j))}
            assert (np.all(order_probs[i][j]["score"] > 0))

            if norm_scores:
                # "score"s are greater zero (> 0), normalize "transition" probabilities, such that the sum of
                # probabilities to get from i -> j (for fixed r and all s) is one.
                order_probs[i][j]["score"] = normalize(order_probs[i][j]["score"], norm="l1", axis=1)
                assert (np.allclose(np.sum(order_probs[i][j]["score"], axis=1), 1.0))

            order_probs[i][j]["log_score"] = np.log(order_probs[i][j]["score"])

        return order_probs

    @staticmethod
    def _get_normalization_constant_Z(marginals, margin_type):
        """
        Calculate the normalization constant Z (see Sections 2.3.1 and 2.3.4). The constant is calculated differently
        depending on the marginal type, i.e. "sum" and "max".

        * "sum": Z_sum = sum r' in {1, ..., n_i} mu_sum(z_i = t' | T)
        
        * "max": Z_max = max r' in {1, ..., n_i} mu_max(z_i = t' | T)   
        
        Here, we can freely choose the i. That is because the margins (mu) are already aggregated over all i' != i.

        :param marginals: dictionary,
            keys: variable;
            values: array-like, shape = (n_i, ), un-normalized marginal value for all assignments

        :param margin_type: string, for which margin type the normalization constant Z should be calculated.

        :return: scalar, constant to normalize the sum (mu_sum) or max (mu_max) marginals
        """
        if isinstance(marginals, dict) or isinstance(marginals, OrderedDict):
            marg_0 = marginals[0]  # we can choose _any_ if the un-normalized marginals to calculate Z
        elif isinstance(marginals, np.ndarray):
            marg_0 = marginals
        else:
            raise ValueError("Marginal(s) must be of type 'dict', 'OrderedDict' or 'ndarray'.")

        if margin_type == "sum":
            Z = np.sum(marg_0)
        elif margin_type == "max":
            Z = np.max(marg_0)
        else:
            raise ValueError("Invalid margin type '%s'. Choices are 'sum' and 'max'")

        return Z

    @staticmethod
    def _normalize_marginals(marginals, margin_type, normalize):
        """
        Normalize the marginals, i.e. mu -> p (see Sections 2.3.1 and 2.3.4)

        :param marginals: dictionary,
            keys: variable;
            values: array-like, shape = (n_i, ), un-normalized marginal value for all assignments

        :param margin_type: string, for which margin type the normalization constant Z should be calculated.

        :param normalize: boolean, indicating whether the margin passed to the function should be normalized. If False,
            the passed margin is not changed.

        :return: normalized marginals: dictionary,
            keys: variable;
            values: array-like, shape = (n_i, ), normalized marginal value for all assignments
        """
        if normalize:
            Z = FactorGraph._get_normalization_constant_Z(marginals, margin_type)
            for i in marginals:
                marginals[i] /= Z
        else:
            pass

        return marginals


class TreeFactorGraph(FactorGraph):
    def __init__(self, candidates, var_conn_graph, make_order_probs, order_probs=None, use_log_space=True,
                 D=0.5, norm_order_scores=False):
        """
        Class to perform inference on tree like a Markov random field (MRF). The sum- and max-marginals can be returned
        using the sum- respectively max-product algorithm. Also the maximum a-posteriori (MAP) estimate can be
        calculated.

        For the implementation of the factor graph and the corresponding inference algorithms we followed:

            [1] MacKay, D. J. "Information theory, inference and learning algorithms", Cambridge university press (2003)
            [2] Bishop, C. "Pattern Recognition and Machine Learning", Springer New York (2006)

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param var_conn_graph: networkx graph, graph representing the variable (ms-feature) connectivity. This graph
            must be tree-like. See "_get_*_connectivity" function in 'RetentionTimeTreeFactorGraph' and
            'RandomTreeFactorGraph'.

        :param make_order_probs: callable, function implementing Sigma of the edge potential function (see Section
            2.2.3). The function should be vectorized, i.e. being able to handle an np.ndarray as input.

        :param order_probs: dict of dicts, storing all pairwise edge potential matrices of size Z_i x Z_j for all pairs
            (i, j) corresponding to the f_rt factors. The matrices store the edge potential values for each candidate
            structure pair of a pair (i, j) (see Section 2.2.3). This parameter can be used if the matrices are pre-
            computed. By default, this parameter is None and the 'make_order_probs' function is used to calculate the
            matrices directly from the given preference values in 'candidates'

        :param use_log_space: boolean, indicating whether the inference should be done in the log-space. That means,
            the edge potential value are logarithmised. Max-product --> Max-sum algorithm. This implementation currently
            only supports the implementation in the log-space.

        :param D: scalar, weight of the retention order information (see Section 2.2.4).

        :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one. (default = False)
        """

        if not use_log_space:
            raise NotImplementedError("Tree factor graphs only support log-space computations.")

        self.var_conn_graph = var_conn_graph

        # Dictionary enabling access to the nodes by their degree
        self.degree_for_var = self.var_conn_graph.degree()
        self.var_for_degree = self._invert_degree_dict(self.degree_for_var)
        self.max_degree = max(self.var_for_degree)  # maximum variable node degree

        # Choose a maximum degree node as root. Ties are broken by choosing the first variable in the variable list.
        self.root = self.var_for_degree[self.max_degree][0]
        LOGGER.debug("Root: %s" % self.root)

        # Create forward- and backward-pass directed trees
        self.di_var_conn_graph_backward = nx.traversal.bfs_tree(self.var_conn_graph, self.root)
        self.di_var_conn_graph_forward = self.di_var_conn_graph_backward.reverse()

        LOGGER.debug("Forward-graph node order: %s" % self.di_var_conn_graph_forward.nodes())
        LOGGER.debug("Backward-graph node order: %s" % self.di_var_conn_graph_backward.nodes())
        LOGGER.debug("%s" % nx.traversal.breadth_first_search.deque(self.di_var_conn_graph_backward))

        # Dictionary enabling access to the neighbors of the variable nodes
        self.var_neighbors = {i: list(neighbors.keys()) for i, neighbors in self.var_conn_graph.adjacency()}

        # Get rt-factors from the MST
        fac_rt = [(src, trg) for (trg, src) in nx.algorithms.traversal.bfs_edges(self.var_conn_graph, source=self.root)]

        super(TreeFactorGraph, self).__init__(candidates=candidates, make_order_probs=make_order_probs,
                                              use_log_space=use_log_space, fac_rt=fac_rt, order_probs=order_probs,
                                              D=D, norm_order_scores=norm_order_scores)

    def __str__(self):
        """
        Return a description of the tree like Markov random field.

        :return: string, describing the tree like MRF
        """
        deg = np.repeat(list(self.var_for_degree.keys()), [len(v) for v in self.var_for_degree.values()])
        rt_diffs = [self.var_conn_graph.get_edge_data(u, v)["rt_diff"] for u, v in self.var_conn_graph.edges]

        return "Root: %d\n" \
               "Degree stats: min=%d, max=%d, avg=%.2f, med=%.2f\n" \
               "Retention time differences: min=%.3f, max=%.3f, avg=%.3f, med=%.3f" % \
               (self.root, min(self.var_for_degree), max(self.var_for_degree), np.mean(deg).item(),
                np.median(deg).item(), min(rt_diffs), max(rt_diffs), np.mean(rt_diffs).item(),
                np.median(rt_diffs).item())

    def _forward_pass(self, aggregation_function) -> (np.array, np.array, np.array, nx.Graph):
        """
        Forward pass of the forward-backward (aka sum- or max-product) algorithm.

        :param aggregation_function: string, either 'sum' or 'max' defining whether the sum- or max-product is used.
        :return: tuple (
            dictionary, containing the R messages
            dictionary, containing the Q messages
            callable, function used for the message aggregation (depending whether sum- or max-product was chosen)
            array-like, accumulated messages at the root of tree like MRF for each candidate
            networkx graph, back-tracking graph for the max-product algorithm, e.g. inference of the MAP solution
        )
        """
        R = OrderedDict()
        Q = OrderedDict()

        # FIXME: For 'sum' we assume working in the log-space!
        if aggregation_function == "sum":
            def agg_fun(x):
                return logsumexp(x, axis=1), None  # sum-product algorithm
            backtracking_graph = None
        elif aggregation_function == "max":
            def agg_fun(x):
                max_idc = np.argmax(x, axis=1)
                max_val = x[np.arange(x.shape[0]), max_idc]
                return max_val, max_idc  # max-product algorithm
            backtracking_graph = nx.Graph()
        else:
            raise ValueError("Invalid aggregation function: '%s'. Chose from 'sum' and 'max'." % aggregation_function)

        q_i__ij = None

        for i in nx.algorithms.traversal.dfs_postorder_nodes(self.var_conn_graph, source=self.root):
            j_src = list(self.di_var_conn_graph_forward.predecessors(i))
            j_trg = list(self.di_var_conn_graph_forward.successors(i))
            LOGGER.debug("Forward: var=%d" % i)
            LOGGER.debug("\tsrc=%s, trg=%s" % (str(j_src), str(j_trg)))

            # Initialize the MS-factor node (R-message)
            R[(i, i)] = {i: self.D_ms * self.candidates[i]["log_score"]}
            LOGGER.debug("\tMS-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                np.min(R[(i, i)][i]), np.max(R[(i, i)][i]), np.mean(R[(i, i)][i]).item(),
                np.median(R[(i, i)][i]).item()))

            # Collect R-messages from neighboring factors to build Q-message
            q_i__ij = R[(i, i)][i]  # ms-factor
            for _j_src in j_src:
                q_i__ij = q_i__ij + R[(_j_src, i)][i]  # rt-factors
            assert (q_i__ij.shape == (self.candidates[i]["n_cand"],))

            if len(j_trg) == 0:
                # No target to send messages anymore. We reached the root.
                assert (i == self.root), "Only the root does not have a further target node."
            else:
                # Still a target to send messages. Have not yet reached the root.
                assert (len(j_trg) == 1), "There should be only one target to send messages to."

                j_trg = j_trg[0]  # outgoing edge
                Q[i] = {(i, j_trg): q_i__ij}

                # gamma_rs's, probabilities of the candidates based on the retention order
                _rt_scores = self.D_rt * self.order_probs[i][j_trg]["log_score"]

                LOGGER.debug("\tRT-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                    np.min(_rt_scores), np.max(_rt_scores), np.mean(_rt_scores).item(), np.median(_rt_scores).item()))

                _tmp = agg_fun(_rt_scores.T + Q[i][(i, j_trg)])
                R[(i, j_trg)] = {j_trg: _tmp[0]}

                LOGGER.debug("\tR-message-score: min=%.3f, max=%.3f, mean=%.3f, med=%.3f" % (
                    np.min(R[(i, j_trg)][j_trg]), np.max(R[(i, j_trg)][j_trg]), np.mean(R[(i, j_trg)][j_trg]).item(),
                    np.median(R[(i, j_trg)][j_trg]).item()))

                if aggregation_function == "max":
                    backtracking_graph.add_edge(j_trg, i, best_candidate=_tmp[1])

        if aggregation_function == "max":
            acc = q_i__ij
        else:
            acc = None

        return R, Q, agg_fun, acc, backtracking_graph

    def _backward_pass(self, R, Q, agg_fun):
        """
        Backward pass of the forward-backward (aka sum- or max-product) algorithm.

        :param R: dictionary, containing the R messages
        :param Q: dictionary, containing the Q messages
        :param agg_fun: callable, function used for the message aggregation (depending whether sum- or max-product was
            chosen)
        :return: tuple (
            dictionary, containing the R messages
            dictionary, containing the Q messages
        )
        """
        for i in nx.algorithms.traversal.dfs_preorder_nodes(self.var_conn_graph, source=self.root):
            j_src = list(self.di_var_conn_graph_backward.predecessors(i))
            j_trg = list(self.di_var_conn_graph_backward.successors(i))
            LOGGER.debug("Backward: var=%d, src=%s, trg=%s" % (i, str(j_src), str(j_trg)))

            assert ((len(j_src) == 0 and i == self.root) or (len(j_src) == 1 and i != self.root))

            # Collect R-messages
            # q-messages going to the MS factor
            Q[i] = OrderedDict([((i, i), np.zeros((self.candidates[i]["n_cand"],)))])
            for j in j_trg:
                Q[i][(i, i)] += R[(j, i)][i]
            for j in j_src:
                Q[i][(i, i)] += R[(i, j)][i]

            # q-messages going to the RT factors
            for _j_trg in j_trg:
                Q[i][(i, _j_trg)] = R[(i, i)][i]
                for j in j_trg:
                    if j == _j_trg:  # collect r message except from the node we send the message
                        continue
                    Q[i][(i, _j_trg)] = Q[i][(i, _j_trg)] + R[(j, i)][i]
                for j in j_src:
                    if j == _j_trg:  # collect r message except from the node we send the message
                        continue
                    Q[i][(i, _j_trg)] = Q[i][(i, _j_trg)] + R[(i, j)][i]

            # gamma_rs's, probabilities of the candidates based on the retention order
            for _j_trg in j_trg:
                _tmp, _ = agg_fun(self.D_rt * self.order_probs[_j_trg][i]["log_score"] + Q[i][(i, _j_trg)])
                if (_j_trg, i) not in R:
                    R[(_j_trg, i)] = {_j_trg: _tmp}
                else:
                    R[(_j_trg, i)][_j_trg] = _tmp
                assert (R[(_j_trg, i)][_j_trg].shape == (self.candidates[_j_trg]["n_cand"],))

        return R, Q

    def sum_product(self):
        """
        Sum product algorithm to prepare for marginal computation.

        :return: self, reference to the object
        """
        # Forward pass & Backward pass with 'sum'
        R, Q, agg_fun, _, _ = self._forward_pass("sum")
        self.R_sum, self.Q_sum = self._backward_pass(R, Q, agg_fun)
        return self

    def max_product(self):
        """
        Max product algorithm to prepare for max-marginal and MAP computation.

        :return: self, reference to the object
        """
        # Forward pass & Backward pass with 'sum'
        R, Q, agg_fun, self.acc_max, self.Par_max = self._forward_pass("max")
        self.R_max, self.Q_max = self._backward_pass(R, Q, agg_fun)
        return self

    def MAP(self):
        """
        Get the Maximum a posteriori estimation (Z_max) and the corresponding lh value (p_max)

        :return: tuple (
            list, length=n_variables, selected candidate for each MS2
            scalar, (log-)likelihood value of Z_max
        )
        """
        assert ((self.acc_max is not None) and
                (self.Par_max is not None)), "Run 'Max-product' first!"

        # Find Z_max via backtracking
        N = len(self.var)
        Z_max = np.full((N,), fill_value=-1, dtype=int)  # i = N
        idx_max = np.argmax(self.acc_max)
        Z_max[self.root] = idx_max
        for j_src, i in nx.algorithms.traversal.dfs_edges(self.Par_max, source=self.root):
            Z_max[i] = self.Par_max.edges[(j_src, i)]["best_candidate"][Z_max[j_src]]
            LOGGER.debug("edge=(%d --> %d), best_candidate=%s, Z_max[%d]=%d" % (
                j_src, i, str(self.Par_max.edges[(j_src, i)]["best_candidate"]), i, Z_max[i]))

        for i in range(N):
            assert (Z_max[i] >= 0), "non-negative candidate indices"
            assert (Z_max[i] < self.candidates[i]["n_cand"])

        # Likelihood of Maximum a posteriori Z_max: p_max
        p_max = self.acc_max[idx_max]
        LOGGER.debug("(Z_max, p_max, p_max (lh-function)): (%s, %f, %f)" % (
            str(Z_max), p_max, self.likelihood(Z_max, log=self.use_log_space)))
        np.testing.assert_allclose(self.likelihood(Z_max, log=self.use_log_space), p_max)

        return Z_max, p_max

    def MAP_only(self):
        """
        Get the Maximum a posteriori estimation (Z_max) and the corresponding lh value (p_max)

        :return: tuple (
            list, length=n_variables, selected candidate for each MS2
            scalar, (log-)likelihood value of Z_max
        )
        """
        _, _, _, self.acc_max, self.Par_max = self._forward_pass("max")
        return self.MAP()

    def MAP_only__brute_force(self):
        """
        Get the Maximum a posteriori estimation (Z_max) and the corresponding lh value (p_max) using brute force. That
        means, we find the MAP estimate simply by enumerating all possible assignments to the discrete random variable.

        :return: tuple (
            list, length=n_variables, selected candidate for each MS2
            scalar, (log-)likelihood value of Z_max
        )
        """
        Z_space = [range(self.candidates[v]["n_cand"]) for v in self.var]

        max_llh = -np.inf
        z_max = None

        for z in it.product(*Z_space):
            llh = self.likelihood(z, log=True)
            if llh > max_llh:
                max_llh = llh
                z_max = z

        if self.use_log_space:
            p_max = max_llh
        else:
            p_max = np.exp(max_llh)

        return z_max, p_max

    def _marginals(self, R) -> OrderedDict:
        """
        Calculate for all variables

        :param R: dictionary, containing the R messages

        :return: dictionary, keys: variable; values: array-like, shape=(n_i,), marginal probabilities for all
            assignments
        """
        marginals = OrderedDict()

        for i in self.var:
            # Collect r-messages on incoming edges
            r_i = R[(i, i)][i]

            r_ij = np.zeros_like(r_i)
            for j in self.var_neighbors[i]:
                # Note: We do not "know" which directions the messages have traveled, i.e. how to access them from
                #       the R dictionary.
                if (i, j) in R:
                    r_ij += R[(i, j)][i]
                elif (j, i) in R:
                    r_ij += R[(j, i)][i]
                else:
                    raise Exception("Whoops")

            marginals[i] = np.exp(r_i + r_ij)  # go from log-space to linear-space

        return marginals

    def get_sum_marginals(self, normalize=True) -> dict:
        """
        Calculate (normalized) sum-marginals for all variables

        :param normalize: boolean indicating, whether the marginal should be normalized (default=True)

        :return: dictionary,
            keys: variable;
            values: array-like, shape = (n_i, ), marginal probabilities for all assignments
        """
        if self.R_sum is None:
            raise RuntimeError("Run 'sum-product' first!")

        return self._normalize_marginals(self._marginals(self.R_sum), "sum", normalize)

    def get_max_marginals(self, normalize=True) -> dict:
        """
        Calculate (normalized) max-marginals for all variables

        :param normalize: boolean indicating, whether the marginal should be normalized (default=True)

        :return: dictionary,
            keys: variable;
            values: array-like, shape = (n_i, ), marginal probabilities for all assignments
        """
        if self.R_max is None:
            raise RuntimeError("Run 'max-product' first!")

        return self._normalize_marginals(self._marginals(self.R_max), "max", normalize)

    @staticmethod
    def _invert_degree_dict(degs) -> OrderedDict:
        """
        Takes the output of nx.Graph.degree() and inverts the dictionary such that all nodes of a certain degree
        can be accessed.

        :param degs: dictionary, keys: node; values: degree, node degrees
        :return: dictionary, keys: degree; values: node, nodes per degree
        """
        degs_out = OrderedDict()
        for node, deg in degs:
            if deg not in degs_out:
                degs_out[deg] = [node]
            else:
                degs_out[deg].append(node)
        return degs_out


class RetentionTimeTreeFactorGraph(TreeFactorGraph):
    def __init__(self, candidates, make_order_probs, order_probs=None, use_log_space=True, min_rt_diff=0.0,
                 D=0.5, norm_order_scores=False):
        """
        Class representing a tree like factor graph derived from the ms-features by using a minimum spanning tree (MST)
        considering a minimum retention time difference between features.

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param make_order_probs: callable, function implementing Sigma of the edge potential function (see Section
            2.2.3). The function should be vectorized, i.e. being able to handle an np.ndarray as input.

        :param order_probs: dict of dicts, storing all pairwise edge potential matrices of size Z_i x Z_j for all pairs
            (i, j) corresponding to the f_rt factors. The matrices store the edge potential values for each candidate
            structure pair of a pair (i, j) (see Section 2.2.3). This parameter can be used if the matrices are pre-
            computed. By default, this parameter is None and the 'make_order_probs' function is used to calcualte the
            matrices directly from the given preference values in 'candidates'

        :param use_log_space: boolean, indicating whether the inference should be done in the log-space. That means,
            the edge potential value are logarithmised. Max-product --> Max-sum algorithm. This implementation currently
            only supports the implementation in the log-space.

        :param D: scalar, weight of the retention order information (see Section 2.2.4).

        :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one. (default = False)
        """

        # Minimum-spanning-tree variable graph
        self.min_rt_diff = min_rt_diff
        self.var_conn_graph = self._get_mst_connectivity(candidates, self.min_rt_diff)

        super(RetentionTimeTreeFactorGraph, self).__init__(candidates=candidates, make_order_probs=make_order_probs,
                                                           order_probs=order_probs, use_log_space=use_log_space,
                                                           D=D, var_conn_graph=self.var_conn_graph,
                                                           norm_order_scores=norm_order_scores)

    @staticmethod
    def _get_mst_connectivity(candidates, min_rt_diff):
        """
        Calculate the connectivity tree (graph) indicating which (i, j)-pairs are used for the MS2 and RT
        integration.

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param min_rt_diff: scalar, minimum RT difference to required for a feature pair to be included in the MST.

        :return: networkx graph
        """
        var_conn_graph = nx.Graph()

        # Add variable nodes
        var = list(candidates.keys())
        for i in var:
            var_conn_graph.add_node(i, retention_time=candidates[i]["retention_time"])

        # Add edges connecting the variable nodes, i.e. pairs considered for the score integration
        for i, j in it.combinations(var, 2):
            rt_diff_ij = var_conn_graph.nodes[j]["retention_time"] - var_conn_graph.nodes[i]["retention_time"]
            assert (rt_diff_ij >= 0)

            if rt_diff_ij < min_rt_diff:
                continue

            var_conn_graph.add_edge(i, j, weight=rt_diff_ij, rt_diff=rt_diff_ij)

        assert (nx.is_connected(var_conn_graph)), "Variable connectivity graph must be connected."

        # Run minimum-spanning-tree algorithm to remove edges and reduce the factor graph to a factor tree
        var_conn_graph = nx.algorithms.tree.minimum_spanning_tree(var_conn_graph)

        return var_conn_graph


class RandomTreeFactorGraph(TreeFactorGraph):
    def __init__(self, candidates, make_order_probs, order_probs=None, use_log_space=True, D=0.5,
                 random_state=None, norm_order_scores=False, remove_edges_with_zero_rt_diff=False):
        """
        Class representing a tree like factor graph derived from the ms-features using random sampling (see
        Section 2.3.2).

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param make_order_probs: callable, function implementing Sigma of the edge potential function (see Section
            2.2.3). The function should be vectorized, i.e. being able to handle an np.ndarray as input.

        :param order_probs: dict of dicts, storing all pairwise edge potential matrices of size Z_i x Z_j for all pairs
            (i, j) corresponding to the f_rt factors. The matrices store the edge potential values for each candidate
            structure pair of a pair (i, j) (see Section 2.2.3). This parameter can be used if the matrices are pre-
            computed. By default, this parameter is None and the 'make_order_probs' function is used to calcualte the
            matrices directly from the given preference values in 'candidates'

        :param use_log_space: boolean, indicating whether the inference should be done in the log-space. That means,
            the edge potential value are logarithmised. Max-product --> Max-sum algorithm. This implementation currently
            only supports the implementation in the log-space.

        :param D: scalar, weight of the retention order information (see Section 2.2.4).

        :param random_state: None | int | instance of RandomState used as seed for the random tree sampling
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        :param norm_order_scores: boolean, indicating whether the the edge potential matrices should be normalized, such
            that the rows sum up to one. (default = False)

        :param remove_edges_with_zero_rt_diff: boolean, indicating whether edges with zero RT difference should be
            included in the random spanning tree.
        """

        self.rs = check_random_state(random_state)
        self.remove_edges_with_zero_rt_diff = remove_edges_with_zero_rt_diff

        super(RandomTreeFactorGraph, self).__init__(
            candidates=candidates, make_order_probs=make_order_probs, order_probs=order_probs,
            use_log_space=use_log_space, D=D, norm_order_scores=norm_order_scores,
            var_conn_graph=self._get_random_connectivity(candidates, self.rs, self.remove_edges_with_zero_rt_diff))

    @staticmethod
    def _get_random_connectivity(candidates, rs, remove_edges_with_zero_rt_diff):
        """
        Sample a random spanning tree from the full MRF.

        :param candidates: dictionary of dicts, containing the candidate set information. 'Keys' are the indices of the
            candidate sets. 'Values' are dictionaries containing the MS-scores, RT, number of candidates, ...

            See documentation of 'prepare_candidate_set_IOKR' and 'prepare_candidate_set_MetFrag' those output is
            expected here.

        :param random_state: None | int | instance of RandomState used as seed for the random tree sampling
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        :param remove_edges_with_zero_rt_diff: boolean, indicating whether edges with zero RT difference should be
            included in the random spanning tree.

        :return: networkx graph
        """
        # Output graph
        var_conn_graph = nx.Graph()

        # Add variable nodes and edges with random weight
        var = list(candidates.keys())
        for i in var:
            var_conn_graph.add_node(i, retention_time=candidates[i]["retention_time"])

        for i, j in it.combinations(var, 2):
            rt_i, rt_j = var_conn_graph.nodes[i]["retention_time"], var_conn_graph.nodes[j]["retention_time"]
            rt_diff_ij = rt_j - rt_i
            assert (rt_diff_ij >= 0)

            if remove_edges_with_zero_rt_diff and (rt_i == rt_j):
                edge_weight = np.inf  # Such edges will not be chosen in the MST
            else:
                edge_weight = rs.rand()

            var_conn_graph.add_edge(i, j, weight=edge_weight, rt_diff=rt_diff_ij)

        # Get minimum spanning tree
        var_conn_graph = nx.algorithms.tree.minimum_spanning_tree(var_conn_graph)

        if remove_edges_with_zero_rt_diff:
            # Test that all edges have retention-time difference larger than 0
            for u, v in var_conn_graph.edges:
                assert (var_conn_graph.get_edge_data(u, v)["rt_diff"] > 0)

        return var_conn_graph


# ----------------------------------------
# Code not used in the paper
# ----------------------------------------
class ChainFactorGraph(FactorGraph):
    def __init__(self, candidates, make_order_probs, order_probs=None, use_log_space=True, D=0.5):
        """
        :param candidates:
        :param make_order_probs:
        :param use_log_space: boolean indicating whether multiplications should be replaced by sum of logs.
            (default=True)
        """
        # Get rt-factors assuming a chain like factor-graph
        fac_rt = [(i, i + 1) for i in range(len(candidates) - 1)]  # (1, 2), (2, 3), ..., (N-1, N)

        super(ChainFactorGraph, self).__init__(candidates=candidates, make_order_probs=make_order_probs,
                                               use_log_space=use_log_space, fac_rt=fac_rt, order_probs=order_probs,
                                               D=D)

    def sum_product(self):
        """
        Sum product algorithm to prepare for marginal computation.

        :param normalize: boolean indicating, whether the marginal should be normalized (default=True)

        :return: dictionary, keys: variable; values: array-like, shape=(n_i,), marginal probabilities for all assignments
        """
        R = {node: {} for node in self.fac_ms + self.fac_rt}
        Q = {node: {} for node in self.var}

        N = len(self.var)

        # === Message propagation (Forward) ===
        # Leaf-factor nodes --> variable nodes
        for i in range(N):  # O(N)
            # theta_ir's, probabilities of the candidates based on MS2
            if self.use_log_space:
                R[(i, i)][i] = self.D_ms * self.candidates[i]["log_score"]  # log(r^T)
            else:
                R[(i, i)][i] = self.candidates[i]["score"] ** self.D_ms  # r^T

        for i in range(N - 1):  # O(N * M_max + N * M_max^2)
            if i == 0:
                Q[i][(i, i + 1)] = R[(i, i)][i]
            else:
                if self.use_log_space:
                    Q[i][(i, i + 1)] = R[(i, i)][i] + R[(i - 1, i)][i]  # log(q) = log(r^T) + log(r^G)
                else:
                    Q[i][(i, i + 1)] = R[(i, i)][i] * R[(i - 1, i)][i]  # q = r^T * r^G

            # gamma_rs's, probabilities of the candidates based on the retention order
            if self.use_log_space:
                R[(i, i + 1)][i + 1] = logsumexp(
                    (self.D_rt * self.order_probs[i][i + 1]["log_score"].T) + Q[i][(i, i + 1)], axis=1)
            else:
                R[(i, i + 1)][i + 1] = (self.order_probs[i][i + 1]["score"].T ** self.D_rt) @ Q[i][(i, i + 1)]

            assert (Q[i][(i, i + 1)].shape == (self.candidates[i]["n_cand"],))
            assert (R[(i, i + 1)][i + 1].shape == (self.candidates[i + 1]["n_cand"],))

        # Message propagation (Backward)
        for i in reversed(range(1, N)):  # O(N * M_max + N * M_max^2)
            if i == (N - 1):
                Q[i][(i, i)] = R[(i - 1, i)][i]
                Q[i][(i - 1, i)] = R[(i, i)][i]
            else:
                if self.use_log_space:
                    Q[i][(i, i)] = R[(i, i + 1)][i] + R[(i - 1, i)][i]
                    Q[i][(i - 1, i)] = R[(i, i)][i] + R[(i, i + 1)][i]
                else:
                    Q[i][(i, i)] = R[(i, i + 1)][i] * R[(i - 1, i)][i]
                    Q[i][(i - 1, i)] = R[(i, i)][i] * R[(i, i + 1)][i]

            # gamma_rs's, probabilities of the candidates based on the retention order
            if self.use_log_space:
                R[(i - 1, i)][i - 1] = logsumexp(
                    (self.D_rt * self.order_probs[i - 1][i]["log_score"]) + Q[i][(i - 1, i)], axis=1)
            else:
                R[(i - 1, i)][i - 1] = (self.order_probs[i - 1][i]["score"] ** self.D_rt) @ Q[i][(i - 1, i)]

            assert (Q[i][(i, i)].shape == (self.candidates[i]["n_cand"],))
            assert (Q[i][(i - 1, i)].shape == (self.candidates[i]["n_cand"],))
            assert (R[(i - 1, i)][i - 1].shape == (self.candidates[i - 1]["n_cand"],))

        # Never accessed
        Q[0][(0, 0)] = R[(0, 1)][0]

        self.R_sum = R
        self.Q_sum = Q

        return self

    def max_product(self):
        """
        Find Z_max maximizing the likelihood of the given factor graph.

        Aka: Viterbi-decoding, Min-Sum algorithm, ...

        :return: tuple (
            list, length=n_variables, selected candidate for each MS2
            scalar, (log-)likelihood value of Z_max
        )
        """

        R = {node: {} for node in self.fac_ms + self.fac_rt}

        # Track parents of nodes along the maximal paths for simple back-tracking
        Par = {}

        N = len(self.var)

        # === Message propagation (Forward) ===
        # Leaf-factor nodes --> variable nodes
        for i in range(N):  # O(N)
            # theta_ir's, probabilities of the candidates based on MS2
            if self.use_log_space:
                R[(i, i)][i] = self.D_ms * self.candidates[i]["log_score"]  # log(r^T)
            else:
                R[(i, i)][i] = self.candidates[i]["score"] ** self.D_ms  # r^T

        for i in range(N - 1):  # O(N * M_max + N * M_max^2)
            if i == 0:
                q_i__iipp = R[(i, i)][i]
            else:
                if self.use_log_space:
                    q_i__iipp = R[(i, i)][i] + R[(i - 1, i)][i]  # log(q) = log(r^T) + log(r^G)
                else:
                    q_i__iipp = R[(i, i)][i] * R[(i - 1, i)][i]  # q = r^T * r^G

            assert (q_i__iipp.shape == (self.candidates[i]["n_cand"],))

            # gamma_rs's, probabilities of the candidates based on the retention order
            if self.use_log_space:
                _tmp = (self.D_rt * self.order_probs[i][i + 1]["log_score"].T) + q_i__iipp
            else:
                _tmp = (self.order_probs[i][i + 1]["score"].T ** self.D_rt) * q_i__iipp

            # Find parent nodes of for layer (i + 1) from layer i using argmax
            Par[i + 1] = np.argmax(_tmp, axis=1)
            # Get the maximum values
            R[(i, i + 1)][i + 1] = _tmp[np.arange(_tmp.shape[0]), Par[i + 1]]

            assert (R[(i, i + 1)][i + 1].shape == (self.candidates[i + 1]["n_cand"],))

        # Accumulate r-messages in last candidate set
        if self.use_log_space:
            acc = R[(N - 1, N - 1)][N - 1] + R[(N - 2, N - 1)][N - 1]
        else:
            acc = R[(N - 1, N - 1)][N - 1] * R[(N - 2, N - 1)][N - 1]

        # Maximum a posteriori: Z_max
        idx_max = np.argmax(acc)
        Z_max = [idx_max]  # i = N
        for i in reversed(range(1, N)):
            Z_max.append(Par[i][Z_max[-1]])
        Z_max.reverse()

        for i in self.var:
            assert (Z_max[i] in range(self.candidates[i]["n_cand"]))

        # Likelihood of Maximum a posteriori Z_max: p_max
        p_max = acc[idx_max]
        np.testing.assert_allclose(self.likelihood(Z_max, log=self.use_log_space), p_max)

        self.Par_max = Par
        self.acc_max = acc
        self.R_max = R
        self.p_max = p_max
        self.Z_max = Z_max

        return self

    def get_sum_marginals(self, normalize=True):
        """
        Calculate (normalized) get_sum_marginals for all variables

        :param normalize: boolean indicating, whether the marginal should be normalized (default=True)
        :return: dictionary, keys: variable; values: array-like, shape=(n_i,), marginal probabilities for all assignments
        """
        marginals = OrderedDict()

        for i in self.var:
            # Collect r-messages on incoming edges
            r_i = self.R_sum[(i, i)][i]

            if self.use_log_space:
                r_immi = self.R_sum[(i - 1, i)][i] if (i - 1, i) in self.R_sum else np.zeros_like(r_i)
                r_iipp = self.R_sum[(i, i + 1)][i] if (i, i + 1) in self.R_sum else np.zeros_like(r_i)

                marg = np.exp(r_immi + r_iipp + r_i)
            else:
                r_immi = self.R_sum[(i - 1, i)][i] if (i - 1, i) in self.R_sum else np.ones_like(r_i)
                r_iipp = self.R_sum[(i, i + 1)][i] if (i, i + 1) in self.R_sum else np.ones_like(r_i)

                marg = r_immi * r_iipp * r_i

            if normalize:
                marg /= np.sum(marg)

            marginals[i] = marg

        return marginals

    def MAP(self):
        return self.Z_max, self.p_max

    def get_max_marginals(self, normalize=True):
        raise NotImplementedError("Max-get_sum_marginals are not implemented yet.")
