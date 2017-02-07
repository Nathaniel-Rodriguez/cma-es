import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class CMAEvolutionaryStrategy:
    """
    Based on code and algorithm from Hansen and DEAP, see below for implementation details:
    Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. https://doi.org/10.1007/11007937_4
    DEAP: https://github.com/DEAP/deap/blob/master/deap/cma.py

    Any functions that are to be optimized MUST be pickleable (serializable). These functions will be minimized.
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
        self._compute_parameters(kwargs)

        # Logging variables
        # Population history is a nested list. The outer list is over generations, the inner list is sorted by
        # performance. Each element is a dictionary with keys={'x', 'fitness'} where 'x' is the member vector
        self.population_history = []
        self.all_time_best = {} # keyed by 'x' and 'fitness'

    def _compute_parameters(self, **kwargs):

        self.sigma = sigma

        ### MIght need effective_sigma back

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

    def _generate_population(self):
        """
        Samples new members from the distribution using the current centroid and covariance matrix
        """

        return self.centroid + self.sigma * self.scaling_of_variables * \
            np.dot(np.random.standard_normal((self.population_size, self.num_of_dimensions)), self.BD.T)

    def _updated_sigma_path(self, c_diff):

        self.sigma_path = (1 - self.sigma_time_const) * self.sigma_path \
            + math.sqrt(self.sigma_time_const * (2 - self.sigma_time_const) * self.mu_effective) / self.sigma \
            * np.dot(self.B, (1. / self.diagD)
                        * np.dot(self.B.T, c_diff))

    def _update_cov_path(self, c_diff, hsig):

        self.cov_matrix_path = (1 - self.cov_time_const) * self.cov_matrix_path + hsig \
            * sqrt(self.cov_time_const * (2 - self.cov_time_const) * self.mu_effective) / self.sigma \
            * c_diff

    def _update_cov_matrix(self, parent_dist_from_centroid, hsig):

        self.covariance_matrix = (1 - self.cov_1 - self.cov_mu + (1 - hsig)
                  * self.cov_1 * self.cov_time_const * (2 - self.cov_time_const)) * self.covariance_matrix \
            + self.cov_1 * numpy.outer(self.cov_matrix_path, self.cov_matrix_path) \
            + self.cov_mu * numpy.dot((self.weights * parent_dist_from_centroid.T), parent_dist_from_centroid) \
            / self.sigma ** 2

    def _update_sigma(self):

        self.sigma *= np.exp((np.linalg.norm(self.sigma_path) / self.chiN - 1.)
                                * self.sigma_time_const / self.sigma_damping)        

    def _update_eigen_decomposition(self):

        self.diagD, self.B = numpy.linalg.eigh(self.covariance_matrix)
        indx = np.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD      

    def _update_log(self, population, fitness_values):
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

    def _update(self, objective_funct, args):
        """
        updates the evolutionary paths, sigma, and the covariance matrix.
        """

        # Sample from distribution to create new offspring and Evaluate fitness values
        population = self._generate_population()
        valid_population, violations = self._boundary_handling()
        fitness_values = Parallel(n_jobs=num_of_jobs)(delayed(objective_funct)(pop, *args) for pop in valid_population )
        corrected_fitness = self._boundary_correction(fitness_values, violations)

        # Sort the results (both values and population)
        fitness_sorted_g2l, population_sorted_by_fitness = zip(*sorted(zip(corrected_fitness, population), reversed=True))
        self._update_log(population_sorted_by_fitness, fitness_sorted_g2l)

        # Calculate the weighted means of the ranked, centroid difference, and h_sig
        old_centroid = self.centroid.copy()
        self.centroid = np.dot(self.weights, population[0:self.num_of_parents])
        c_diff = self.centroid - old_centroid
        hsig = float((np.linalg.norm(self.sigma_path) /
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

    def _nearest_valid_parameters(self, invalid_parameters, bounds):
        """
        Calculates the set of nearest valid parameters given some
        invalid set and the bounds. Returns a valid parameter set
        and the distance between the new and old set, and a flag
        that states whether any invalid parameters were found

        The flag is to avoid float point equivalence comparisons
        """

        valid_parameters = invalid_parameters.copy()
        is_valid = True
        for i in range(self.num_of_dimensions):
            if invalid_parameters[i] < bounds[i][0]:
                valid_parameters[i] = bounds[i][0]
                is_valid = False
            elif invalid_parameters[i] > bounds[i][1]:
                valid_parameters[i] = bounds[i][1]
                is_valid = False

        return valid_parameters, np.linalg.norm(valid_parameters - invalid_parameters), is_valid

    def _boundary_handling(self, population):
        """
        Check population for invalide members.
        Revise member by placing in valid region.
        Return a list of revised population, a list of tuples with indexes of violated members
        and the distances between the invalid and validified members
        """

        if self.bounds == None:
            return population, [ (False,0) for i in range(self.population_size) ]

        violations = []
        valid_population = population.copy()
        for i, member in enumerate(population):
            valid_member, distance, validity = self._nearest_valid_parameters(member, self.bounds)
            valid_population[i] = valid_member
            violations.append((validity, distance))

        return valid_population, violations

    def _boundary_correction(self, fitness, violations):
        """
        Apply corrections to the fitness of members that violated conditions

        This is just an ad hoc adjustment, will implement the following in the 
        future: Errata/Addenda for A Method for Handling
        Uncertainty in Evolutionary Optimization With an
        Application to Feedback Control of Combustion, Nikolaus Hansen, 2011
        """

        corrected_fitness = []
        for i in range(self.population_size):
            if violations[i][0]:
                corrected_fitness.append(fitness[i] + 
                    violations[i][1] * np.median(fitness) / (self.sigma * self.sigma * np.mean(covariance_matrix)))
            else:
                corrected_fitness.append(fitness[i])

        return corrected_fitness

    def engage(self, objective_funct, args, iterations = 100, parallel=True, 
        num_of_jobs=-2, bounds=None, verbose=False):
        """
        Run the update process on the objective function for designated number of iterations.
        If num_of_jobs = -2, one less than max number of processes are run
        If num_of_jobs = -1, max number of processes are run
        If num_of_jobs = 1, it will be run in serial.

        bounds: list of tuples of (min,max) values for each parameter
        """

        self.bounds = bounds
        if parallel:
            self.num_of_jobs = num_of_jobs
        else:
            self.num_of_jobs = 1

        for i in range(iterations):
            if verbose:
                print("Generation: ", i)
            self._update(objective_funct, args)

def fmin(objective_funct, x0, sigma0, args=(), iterations=1000, parallel=True, num_of_jobs=-2, 
    cma_params={}, bounds=[], verbose=False):
    """
    A functional version of the CMA evolutionary strategy

    bounds: a list of tuples in the order of x0. Tuple are the ranges of the parameters accepted.
    """

    cma_object = CMAEvolutionaryStrategy(x0, sigma0, **cma_params)
    cma.engage(objective_funct, args, iterations, parallel, num_of_jobs, bounds, verbose)

if __name__ == '__main__':
    """
    testing
    """

    pass