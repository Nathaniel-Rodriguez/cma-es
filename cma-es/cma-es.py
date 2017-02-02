import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class CMAEvolutionaryStrategy:
    """
    Based on code and algorithm from Hansen, see below for implementation details:
    Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. https://doi.org/10.1007/11007937_4

    Any functions that are to be optimized MUST be pickleable (serializable).
    """

    def __init__(self, x0, sigma0, **kwargs):
        """
        x0 : the initial centroid vector
        sigma0 : the initial step-size (standard deviation)

        The problem variables should have been scaled, such that a single 
        standard deviation on all variables is useful and the optimum is 
        expected to lie within about x0 +- 3*sigma0. (sigma=standard_deviation)

        Can pass an array scaling_of_variables the designates the scale for
        each sigma for each variable independently, else assumed they are
        all the same scale for sigma.
        """

        self.centroid = np.array(x0)
        self.num_of_dimensions = len(x0)
        self.scaling_of_variables = kwargs.get('scaling_of_variables', np.ones(self.num_of_dimensions))
        self.covariance_matrix = kwargs.get('covariance_matrix', np.identity(self.num_of_dimensions))
        self.population_size = kwargs.get('population_size', 4 + math.floor(3 * math.log(self.num_of_dimensions)))
        self.num_of_parents = kwargs.get('num_of_parents', self.population_size / 2.)

        self.compute_parameters(kwargs)

    def compute_parameters(self, **kwargs):

        self.effective_sigma = self.sigma0 * self.scaling_of_variables

        # Generate initial evolutionary paths
        self.cov_matrix_path = np.zeros(self.num_of_dimensions)
        self.sigma_path = np.zeros(self.num_of_dimensions)

        # Generate initial weights and normalize them
        self.weights = math.log(self.num_of_parents + 0.5) - np.log(np.arange(1, self.num_of_parents + 1))
        self.weights /= sum(self.weights)
        # Variance-effectiveness of weights
        self.mu_effective = 1. / sum(self.weights ** 2)

        # Evaluate Covariance matrix decomposition
        self.diagD, self.diagB = np.linalg.eigh(self.covariance_matrix)

        # Generate time-constants
        self.cov_time_const = (4.0 + self.mu_effective / self.num_of_dimensions) / \
            (self.num_of_dimensions + 4 + 2 * self.mu_effective / self.num_of_dimensions)
        self.sigma_time_const = (self.mu_effective + 2) / (self.num_of_dimensions + self.mu_effective + 5)

        # Rank 1 update learning rate for covariance matrix
        self.cov_1 = 2.0 / ((self.num_of_dimensions + 1.3)**2 + self.mu_effective)

        # Rank mu update learning rate for covariance matrix
        self.cov_mu = min([1 - self.cov_1, \
            2 * (self.mu_effective - 2 + 1.0 / self.mu_effective) / ((self.num_of_dimensions + 2)**2 + self.mu_effective)])

        # Damping term for sigma
        self.sigma_damping = 1. + 2 * max(0, math.sqrt((self.mu_effective - 1) / (self.num_of_dimensions + 1)) - 1) + self.sigma_time_const

    def update():

    def eval():

        fitness_values = Parallel(n_jobs=num_of_jobs)(delayed(objective_funct)(args) for i in range(self.population_size) )

    def engage(self, objective_funct, args, iterations = 100, parallel=True, num_of_jobs=-2):
        """
        Run the update process on the objective function for designated number of iterations.
        If num_of_jobs = -2, one less than max number of processes are run
        If num_of_jobs = -1, max number of processes are run
        If num_of_jobs = 1, it will be run in serial.
        """

        pass

if __name__ == '__main__':
    """
    testing
    """

    pass