import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class CMAEvolutionaryStrategy:
    """
    Based on code and algorithm from Hansen and DEAP, see below for implementation details:
    Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. https://doi.org/10.1007/11007937_4
    DEAP: https://github.com/DEAP/deap/blob/master/deap/cma.py

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
        self.sigma0 = sigma0
        self.num_of_dimensions = len(x0)
        self.scaling_of_variables = kwargs.get('scaling_of_variables', np.ones(self.num_of_dimensions))
        self.covariance_matrix = kwargs.get('covariance_matrix', np.identity(self.num_of_dimensions))
        self.population_size = kwargs.get('population_size', 4 + math.floor(3 * math.log(self.num_of_dimensions)))
        self.num_of_parents = kwargs.get('num_of_parents', self.population_size / 2.)
        self.update_count = 0
        self.compute_parameters(kwargs)

        # Logging variables
        # Population history is a nested list. The outer list is over generations, the inner list is sorted by
        # performance. Each element is a dictionary with keys={'x', 'fitness'} where 'x' is the member vector
        self.population_history = []
        self.all_time_best = {} # keyed by 'x' and 'fitness'

    def compute_parameters(self, **kwargs):

        self.effective_sigma = self.sigma0 * self.scaling_of_variables
        self.chiN = math.sqrt(self.dim) * (1 - 1. / (4. * self.dim) + 1. / (21. * self.dim ** 2))
        # Generate initial evolutionary paths
        self.cov_matrix_path = np.zeros(self.num_of_dimensions)
        self.sigma_path = np.zeros(self.num_of_dimensions)

        # Generate initial weights and normalize them
        self.weights = math.log(self.num_of_parents + 0.5) - np.log(np.arange(1, self.num_of_parents + 1))
        self.weights /= sum(self.weights)
        # Variance-effectiveness of weights
        self.mu_effective = 1. / sum(self.weights ** 2)

        # Decompose Covariance matrix
        self._update_eigen_decomposition()

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

    def generate_population(self):
        """
        Samples new members from the distribution using the current centroid and covariance matrix
        """

        return self.centroid + self.effective_sigma * \
            np.dot(np.random.standard_normal((self.population_size, self.num_of_dimensions)), self.BD.T)

    def _updated_sigma_path(self, c_diff):

        self.sigma_path = (1 - self.sigma_time_const) * self.sigma_path \
            + math.sqrt(self.sigma_time_const * (2 - self.sigma_time_const) * self.mu_effective) / self.effective_sigma \
            * np.dot(self.B, (1. / self.diagD)
                        * np.dot(self.B.T, c_diff))

    def _update_cov_path(self, c_diff, hsig):

        self.cov_matrix_path = (1 - self.cov_time_const) * self.cov_matrix_path + hsig \
            * sqrt(self.cov_time_const * (2 - self.cov_time_const) * self.mu_effective) / self.effective_sigma \
            * c_diff

    def _update_cov_matrix(self, parent_dist_from_centroid, hsig):

        self.covariance_matrix = (1 - self.cov_1 - self.cov_mu + (1 - hsig)
                  * self.cov_1 * self.cov_time_const * (2 - self.cov_time_const)) * self.covariance_matrix \
            + self.cov_1 * numpy.outer(self.cov_matrix_path, self.cov_matrix_path) \
            + self.cov_mu * numpy.dot((self.weights * parent_dist_from_centroid.T), parent_dist_from_centroid) \
            / self.effective_sigma ** 2

    def _update_sigma(self):

        self.effective_sigma *= np.exp((np.linalg.norm(self.sigma_path) / self.chiN - 1.)
                                * self.sigma_time_const / self.sigma_damping)        

    def _update_eigen_decomposition(self):

        self.diagD, self.B = numpy.linalg.eigh(self.covariance_matrix)
        indx = np.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD      

    def update(self):
        """
        """

        # Sample from distribution to create new offspring and Evaluate fitness values
        population = generate_population()
        fitness_values = Parallel(n_jobs=num_of_jobs)(delayed(objective_funct)(pop, *args) for pop in population )

        # Sort the results (both values and population)
        fitness_sorted_g2l, population_sorted_by_fitness = zip(*sorted(zip(fitness_values, population), reversed=True))
        self.update_log(population_sorted_by_fitness, fitness_sorted_g2l)

        # Calculate the weighted means of the ranked, centroid difference, and h_sig
        old_centroid = self.centroid.copy()
        self.centroid = np.dot(self.weights, population[0:self.num_of_parents])
        c_diff = self.centroid - old_centroid
        hsig = float((np.linalg.norm(self.sigma_path) /# might get broken if sigma path is a vector
                      math.sqrt(1. - (1. - self.sigma_time_const) ** (2. * (self.update_count + 1.))) / self.chiN
                      < (1.4 + 2. / (self.dim + 1.))))
        self.update_count += 1

        # Update the evolutionary paths
        self._updated_sigma_path(c_diff)
        self._update_cov_path(c_diff, hsig)

        # Update the covariance matrix and sigma
        parent_dist_from_centroid = population[:self.num_of_parents] - old_centroid
        self._update_cov_matrix(parent_dist_from_centroid, hsig)
        self._update_sigma()

        # Update eigen decomposition
        self._update_eigen_decomposition()

    def boundary_handling():
        """
        """

        pass

    def eval():

        pass

    def engage(self, objective_funct, args, iterations = 100, parallel=True, num_of_jobs=-2):
        """
        Run the update process on the objective function for designated number of iterations.
        If num_of_jobs = -2, one less than max number of processes are run
        If num_of_jobs = -1, max number of processes are run
        If num_of_jobs = 1, it will be run in serial.
        """

        pass

    def update_log(self, population, fitness_values):
        """
        Adds the new population and fitness values to the population_history
        """

        self.population_history.append({ 'x' : population[i], 'fitness': fitness_values[i]} for i in range(self.population_size) )

        if len(self.all_time_best) == 0:
            self.all_time_best['x'] = self.population_history[-1][0]['x']
            self.all_time_best['fitness'] = self.population_history[-1][0]['fitness']
        elif self.all_time_best['fitness'] < self.population_history[-1][0]['fitness']:
            self.all_time_best['x'] = self.population_history[-1][0]['x']
            self.all_time_best['fitness'] = self.population_history[-1][0]['fitness']

def fmin(objective_funct, x0, sigma0, args=(), iterations=1000, parallel=True, num_of_jobs=-2, cma_params={}, bounds=[]):
    """
    A functional version of the CMA evolutionary strategy

    bounds: a list of tuples in the order of x0. Tuple are the ranges of the parameters accepted.
    """

    pass

if __name__ == '__main__':
    """
    testing
    """

    pass