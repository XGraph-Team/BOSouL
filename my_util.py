import networkx as nx
import cosasi
import torch
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model, fit_fully_bayesian_model_nuts
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import ndlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import numpy as np
from itertools import permutations, combinations

from graphGeneration import connSW, BA, ER, CiteSeer, Cora, PubMed, photo, coms
import random
import math

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from sklearn.cluster import KMeans, SpectralClustering
import statistics as s

################################################
# Global parameters
################################################
diffusion_model = "sir"
infect_rate = 0.1
graph_size = 1000
candidate_size = 50
seed_size = 3
actual_time_step_size = 10
num_iterations = 50
recovery_rate = 0.1

from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from gpytorch.models.exact_gp import ExactGP
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import RBFKernel, RFFKernel
from typing import Any, List, NoReturn, Optional, Union
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior

def get_gaussian_likelihood_with_gamma_prior(
    batch_shape: Optional[torch.Size] = None,
) -> GaussianLikelihood:
    r"""Constructs the GaussianLikelihood that is used by default by
    several models. This uses a Gamma(1.1, 0.05) prior and constrains the
    noise level to be greater than MIN_INFERRED_NOISE_LEVEL (=1e-4).
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    return GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(
            1e-4,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )

# define a class inherited from SingleTaskGP for RBF/rff kernel
class RBFSingleTaskGP(SingleTaskGP):

    def __init__(self, train_X: torch.Tensor,
        train_Y: torch.Tensor,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `RBF`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, ignore_X_dims=ignore_X_dims
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=self._aug_batch_shape
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = RBFKernel()
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        self.covar_module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# define a class inherited from StaticNetworkContagion
# by Zonghan
class Contagion(cosasi.StaticNetworkContagion):

    def __init__(self, G, model=diffusion_model, infection_rate=0.1, recovery_rate=0.1,
                 fraction_infected=None, source=None):

        self.model = model.lower()

        if isinstance(G, nx.classes.graph.Graph):
            self.G = G
        else:
            raise ValueError('G must be a NetworkX instance')

        if isinstance(infection_rate, float) and 0.0 <= infection_rate <= 1.0:
            self.infection_rate = infection_rate
        else:
            raise ValueError("Infection rate must be a float between 0 and 1.")

        if not recovery_rate or (
            isinstance(recovery_rate, float) and 0.0 <= recovery_rate <= 1.0
        ):
            self.recovery_rate = recovery_rate
        else:
            raise ValueError("Recovery rate must be a float between 0 and 1.")

        if fraction_infected and source:
            raise ValueError("User can only provide one of fraction_infected, source.")
        elif not fraction_infected and not source:
            self.fraction_infected = fraction_infected
        else:
            self.fraction_infected = fraction_infected
            self.source = source

        self.history = []

        config = mc.Configuration()
        config.add_model_parameter("beta", self.infection_rate)

        if self.fraction_infected:
            config.add_model_parameter("fraction_infected", self.fraction_infected)
        elif self.source:
            config.add_model_initial_configuration("Infected", self.source)
        else:
            raise NotImplementedError

        if self.model == 'si':
            self.sim = ep.SIModel(graph=self.G)
        elif self.model == 'sir':
            self.sim = ep.SIRModel(graph=self.G)
            if not self.recovery_rate:
                raise ValueError("Recovery rate must be defined for SIR model.")
            config.add_model_parameter("gamma", self.recovery_rate)
        elif self.model == "sis":
            self.sim = ep.SISModel(graph=self.G, seed=self.seed)
            if not self.recovery_rate:
                raise ValueError("Recovery rate must be defined for SIS model.")
            config.add_model_parameter("lambda", self.recovery_rate)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        self.sim.set_initial_status(config)

        return None


# simulate the contagion process and return the peak mean and peak variance of a candidate source set
# by Zonghan
def source_coverage(contagion, c_star, time_step=2 * actual_time_step_size, num_of_sims=10):

    n = contagion.G.number_of_nodes()

    peaks = []

    for iter in range(num_of_sims):
        contagion.reset_sim()
        contagion.forward(time_step)

        peak = 0

        for i in range(time_step):
            subgraph = contagion.get_infected_subgraph(step=i)
            c = list(subgraph.nodes)

            res = len(set(c) & set(c_star))
            coverage = (n - len(c) - len(c_star) + 2 * res) / n
            if coverage >= peak:
                peak = coverage
            if coverage < peak:
                break

        peaks.append(peak)

    peak_mean = torch.tensor(peaks).mean()
    peak_var = torch.tensor(peaks).var()

    return peak_mean, peak_var


# find the top 100 nodes with highest degree centrality, then sample 3 nodes from them as the GT source set
# by Zonghan
def create_true_source_set(G, num_of_sources=3):
    # Todo filter by centralities
    # https://networkx.org/documentation/stable/reference/algorithms/centrality.html

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = deg[:100]

    set = random.sample(candidates, num_of_sources)
    source_set = []
    for item in set:
        source_set.append(item[0])

    return source_set



# top candidate_size nodes with highest degree centrality as candidate pool
# by Zonghan

def create_candidate_pool(G, c_star, candidate_size=50):
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidate_source_nodes = []
    for item in deg:
        a = item[0]
        if a in c_star:
            candidate_source_nodes.append(a)
        if len(candidate_source_nodes) == candidate_size:
            break

    return candidate_source_nodes

def create_candidate_pool_from_whole_graph(G, candidate_size=100):
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    cadidates = []
    for item in deg:
        cadidates.append(item[0])
        if len(cadidates) == candidate_size:
            break
    return cadidates

def sample_from_candidate_pool(candidate_source_nodes, estimated_source_number):
    return random.sample(candidate_source_nodes, estimated_source_number)

# not used in the final version

def sample_from_infected_graph(G, c_star, estimated_source_number):

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidate_source_nodes = []
    for item in deg:
        a = item[0]
        if a in c_star:
            candidate_source_nodes.append(a)
        if len(candidate_source_nodes) == candidate_size:
            break

    return random.sample(candidate_source_nodes, estimated_source_number), candidate_source_nodes

# GNN used as BO surrogate function

class regGCN(torch.nn.Module):
    num_outputs = 1
    def __init__(self):
        super(regGCN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)
        self.lin1 = nn.Linear(1, 16)
        self.lin2 = nn.Linear(16, 8)
        self.lin3 = nn.Linear(8, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = global_mean_pool(x,batch)

        return x

#evaluation metircs

def node_set_distance(s1, s2, G):

    perm_scores = {}

    for s2_perm in permutations(s2):
        perm_scores[s2_perm] = 0
        for i in range(min(len(s1), len(s2))):
            perm_scores[s2_perm] += nx.shortest_path_length(
                G, source=s1[i], target=s2_perm[i]
            )
        # if len(s1) > len(s2):
        #     for j in range(i, len(s1)):
        #         min_add = np.inf
        #         for s in s2_perm:
        #             d = nx.shortest_path_length(G, source=s1[j], target=s)
        #             if d < min_add:
        #                 min_add = d
        #         perm_scores[s2_perm] += min_add
        if len(s2) > len(s1):
            for j in range(i, len(s2_perm)):
                min_add = np.inf
                for s in s1:
                    d = nx.shortest_path_length(G, source=s2_perm[j], target=s)
                    if d < min_add:
                        min_add = d
                perm_scores[s2_perm] += min_add
    return min(perm_scores.values())

# graph sampling

def distance_sampling(s1, ss2, G):

    scores = []

    for s2 in ss2:

        perm_scores = {}

        for s2_perm in permutations(s2):
            perm_scores[s2_perm] = 0
            for i in range(min(len(s1), len(s2))):
                perm_scores[s2_perm] += pow(nx.shortest_path_length(
                    G, source=s1[i], target=s2_perm[i]
                ), 2)
            if len(s1) > len(s2):
                for j in range(i, len(s1)):
                    min_add = np.inf
                    for s in s2_perm:
                        d = nx.shortest_path_length(G, source=s1[j], target=s)
                        if d < min_add:
                            min_add = d
                    perm_scores[s2_perm] += pow(min_add, 2)
            if len(s2) > len(s1):
                for j in range(i, len(s2_perm)):
                    min_add = np.inf
                    for s in s1:
                        d = nx.shortest_path_length(G, source=s2_perm[j], target=s)
                        if d < min_add:
                            min_add = d
                    perm_scores[s2_perm] += pow(min_add, 2)

        score = min(perm_scores.values())
        scores.append(score)

    return min(scores)


################################################
# Sampling based on graph fourier transform
################################################

def fourier_sampler(G, candidates, train_X_fourier, UT, size):

    scores = []

    for candidate in candidates:

        input_for_fourier = []
        for item in G.nodes:
            if item in candidate:
                input_for_fourier.append(1)
            else:
                input_for_fourier.append(0)

        candidate_fourier = np.matmul(input_for_fourier, UT)
        distances = []
        for train in train_X_fourier:
            distance = math.dist(train, candidate_fourier)
            distances.append(distance)
        min_distance = min(distances)
        scores.append(min_distance)

    scores = np.array(scores)
    indices = np.argpartition(scores, -size)[-size:]

    final_candidates = []
    for index in indices:
        final_candidates.append(candidates[index])

    return final_candidates

def fourier_transfer_for_all_candidate_set(candidates, number_of_sources, UT):

    n = len(UT)

    signals = []
    for source_set in combinations(candidates, number_of_sources):
        a = [0 for i in range(n)]
        for node in source_set:
            a[node] = 1
        signal = np.matmul(a, UT)
        signals.append(signal)

    return signals

def find_source_set_from_fourier(signal, number_of_sources, UT_inv):

    source_set = []

    a = np.matmul(signal, UT_inv)
    b = np.around(a)
    for i in range(len(b)):
        if b[i] == 1:
            source_set.append(i)

    if len(source_set) != number_of_sources:
        raise NameError('length of source set is not the estimated number')

    return source_set
