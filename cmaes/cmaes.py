import math
import numpy as np
from joblib import Parallel, delayed

class CMAEvolutionaryStrategy:
    """
    Based on code and algorithm from Hansen and DEAP, see below for 
    implementation details:
    Hansen, N. (2016). The CMA evolution strategy: A tutorial. 
    arXiv:1604.00772. https://doi.org/10.1007/11007937_4
    DEAP: https://github.com/DEAP/deap/blob/master/deap/cma.py

    Any functions that are to be optimized MUST be pickleable (serializable). 
    These functions will be minimized.

    From Hansen on parameter scaling:
    The specific formulation of a (real) optimization problem has a tremendous
    impact on the optimization performance. In particular, a reasonable 
    parameter encoding is essential. All parameters should be rescaled such 
    that they have presumably similar sensitivity (this makes the identity as 
    initial covariance matrix the right choice). Usually, the best approach is
    to write a wrapper around the objective function that transforms 
    the parameters before the actual function call. The wrapper scales, 
    for example, in each parameter/coordinate the value [0; 10] into the 
    typical actual domain of the parameter/coordinate. To achieve this 
    on-the-fly, a linear scaling of variables is provided in the Scilab and 
    Python codes below. With this transformation, a typical initial sigma 
    will be ≈ 2, see also below. The natural encoding of (some of) the 
    parameters can also be "logarithmic". That is, for a parameter that must 
    always be positive, with a ratio between typical upper and lower value 
    being larger than 100, we might use 10x instead of x to call the 
    objective function. More specifically, to achieve the parameter range 
    [10–4,10–1], we use 10–4×103x/10 with x in [0; 10]. Again, the idea is to 
    have similar sensitivity: this makes sense if we expect the change from 
    10–4 to 10–3 to have an impact similar to the change from 10–2 to 10–1. 
    In order to avoid the problem that changes of very small values have too 
    less an impact, an alternative is to choose 10–1 × (x/10)2 ≥ 0. In the 
    case where only a lower bound at zero is necessary, a simple and natural 
    transformation is x2 × default_x, such that x=1 represents the default 
    (or initial) value and x remains unbounded during optimization.

    In summary, to map the values [0;10] into [a;b] we have the alternative 
    transformations :
    a + (b-a) × x/10 or a + (b-a) × (x/10)2 ≥ a or a × (b/a)x/10 ≥ 0.

    NOTE: Only linear scaling is available. Take Hansen's advice and
    make a wrapper to scale yourself between [0,1] and choose an 
    appropriate sigma (0.2)

    """

    def __init__(self, x0, sigma0, **kwargs):
        """
        x0 : the initial centroid vector
        sigma0 : the initial step-size (standard deviation)

        The problem variables should have been scaled, such that a single 
        standard deviation on all variables is useful and the optimum is 
        expected to lie within about x0 +- 3*sigma0. 
        (sigma=standard_deviation)

        Can pass an array scaling_of_variables the designates the scale for
        each sigma for each variable independently, else assumed they are
        all the same scale for sigma.
        """

        self.centroid = np.array(x0)
        self.sigma0 = sigma0
        self.num_of_dimensions = len(x0)
        self.scaling_of_variables = kwargs.get('scaling_of_variables', 
                                            np.ones(self.num_of_dimensions))
        self.covariance_matrix = kwargs.get('covariance_matrix', 
                                        np.identity(self.num_of_dimensions))
        self.population_size = kwargs.get('population_size', 
                        4 + math.floor(3 * math.log(self.num_of_dimensions)))
        self.num_of_parents = kwargs.get('num_of_parents', 
                                            int(self.population_size / 2.))
        self.update_count = 0
        self._compute_parameters(kwargs)

        # Logging variables
        # Population history is a nested list. The outer list is over 
        # generations, the inner list is sorted by performance. 
        # Each element is a dictionary with keys={'x', 'cost'} 
        # where 'x' is the member vector
        self.population_history = []
        # best is keyed by 'x' and 'cost'
        self.all_time_best = {}

    def _compute_parameters(self, kwargs):

        self.sigma = self.sigma0

        self.chiN = math.sqrt(self.num_of_dimensions) * \
            (1 - 1. / (4. * self.num_of_dimensions) + 1. / \
                (21. * self.num_of_dimensions ** 2))
        # Generate initial evolutionary paths
        self.cov_matrix_path = np.zeros(self.num_of_dimensions)
        self.sigma_path = np.zeros(self.num_of_dimensions)

        # Generate initial weights and normalize them
        self.weights = math.log(self.num_of_parents + 0.5) - \
                        np.log(np.arange(1, self.num_of_parents + 1))
        self.weights /= sum(self.weights)
        # Variance-effectiveness of weights
        self.mu_effective = 1. / sum(self.weights ** 2)

        # Decompose Covariance matrix
        self._update_eigen_decomposition()

        # Generate time-constants
        self.cov_time_const = kwargs.get('cov_time_const', 
            4. / (self.num_of_dimensions + 4))
            # (4.0 + self.mu_effective / \
            # self.num_of_dimensions) / (self.num_of_dimensions + 4 + 2 * \
            # self.mu_effective / self.num_of_dimensions)
        self.sigma_time_const = kwargs.get('sigma_time_const', 
            (self.mu_effective + 2.) \
                / (self.num_of_dimensions + self.mu_effective + 3))
            # (self.mu_effective + 2) / \
            # (self.num_of_dimensions + self.mu_effective + 5)

        # Rank 1 update learning rate for covariance matrix
        self.cov_1 = kwargs.get("cov_1", 
            2.0 / ((self.num_of_dimensions + 1.3)**2 \
            + self.mu_effective))

        # Rank mu update learning rate for covariance matrix
        self.cov_mu = kwargs.get("cov_mu", 
            min([1 - self.cov_1, 
                2. * (self.mu_effective - 2. + 1. / self.mu_effective) \
                / ((self.num_of_dimensions + 2.)**2 + self.mu_effective)]))

        # Damping term for sigma
        self.sigma_damping = kwargs.get("sigma_damping", 
            1. + 2 * max(0, math.sqrt((self.mu_effective - 1) \
            / (self.num_of_dimensions + 1)) - 1) + self.sigma_time_const)

    def _generate_population(self):
        """
        Samples new members from the distribution using the current centroid 
        and covariance matrix
        """

        return self.centroid + self.sigma * self.scaling_of_variables * \
            np.dot(np.random.standard_normal((self.population_size, \
                self.num_of_dimensions)), self.BD.T)

    def _updated_sigma_path(self, c_diff):

        self.sigma_path = (1 - self.sigma_time_const) * self.sigma_path \
            + math.sqrt(self.sigma_time_const * (2 - self.sigma_time_const) *\
                self.mu_effective) / self.sigma * \
                np.dot(self.B, (1. / self.diagD) \
                    * np.dot(self.B.T, c_diff))

    def _update_cov_path(self, c_diff, hsig):

        self.cov_matrix_path = (1 - self.cov_time_const) * \
            self.cov_matrix_path + hsig \
            * math.sqrt(self.cov_time_const * (2 - self.cov_time_const) \
            * self.mu_effective) / self.sigma * c_diff

    def _update_cov_matrix(self, parent_dist_from_centroid, hsig):

        self.covariance_matrix = (1 - self.cov_1 - self.cov_mu + (1 - hsig)
                  * self.cov_1 * self.cov_time_const * \
                  (2 - self.cov_time_const)) * self.covariance_matrix \
            + self.cov_1 * np.outer(self.cov_matrix_path, \
                self.cov_matrix_path) + self.cov_mu * np.dot((self.weights * \
                    parent_dist_from_centroid.T), parent_dist_from_centroid) \
            / self.sigma ** 2

    def _update_sigma(self):

        self.sigma *= np.exp((np.linalg.norm(self.sigma_path) / \
            self.chiN - 1.) * self.sigma_time_const / self.sigma_damping)        

    def _update_eigen_decomposition(self):

        self.diagD, self.B = np.linalg.eigh(self.covariance_matrix)
        indx = np.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def _update_log(self, population, cost_values):
        """
        Adds the new population and cost values to the population_history
        """

        self.population_history.append([{ 'x' : population[i], \
            'cost': cost_values[i]} for i in range(self.population_size)] )

        if len(self.all_time_best) == 0:
            self.all_time_best['x'] = self.population_history[-1][0]['x']
            self.all_time_best['cost'] = \
                self.population_history[-1][0]['cost']
        elif self.all_time_best['cost'] < \
            self.population_history[-1][0]['cost']:

            self.all_time_best['x'] = self.population_history[-1][0]['x']
            self.all_time_best['cost'] = \
                self.population_history[-1][0]['cost']

    def _update(self, objective_funct, args):
        """
        updates the evolutionary paths, sigma, and the covariance matrix.
        """

        # Sample from distribution to create new offspring
        population = self._generate_population()
        valid_population, violations = self._boundary_handling(population)
        cost_values = Parallel(n_jobs=self.num_of_jobs)(delayed(\
            objective_funct)(pop, *args) for pop in valid_population )
        corrected_cost = self._boundary_correction(cost_values, violations)

        # Sort the results (both values and population)
        cost_sorted_l2g, population_sorted_by_cost = \
            mutual_sort(corrected_cost, population)
        self._update_log(population_sorted_by_cost, cost_sorted_l2g)

        # Calculate weighted means of ranked, centroid difference, and h_sig
        old_centroid = self.centroid.copy()
        self.centroid = np.dot(self.weights, \
            population_sorted_by_cost[0:self.num_of_parents])
        c_diff = self.centroid - old_centroid
        hsig = float((np.linalg.norm(self.sigma_path) /
                      math.sqrt(1. - (1. - self.sigma_time_const) ** (2. * \
                        (self.update_count + 1.))) / self.chiN
                      < (1.4 + 2. / (self.num_of_dimensions + 1.))))
        self.update_count += 1

        # Update the evolutionary paths
        self._updated_sigma_path(c_diff)
        self._update_cov_path(c_diff, hsig)

        # Update the covariance matrix and sigma
        parent_dist_from_centroid = \
            population_sorted_by_cost[:self.num_of_parents] - old_centroid
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

        return valid_parameters, \
            np.linalg.norm(valid_parameters - invalid_parameters), \
            is_valid

    def _boundary_handling(self, population):
        """
        Check population for invalide members.
        Revise member by placing in valid region.
        Return a list of revised population, a list of tuples with indexes of 
        violated members and the distances between the invalid and 
        validified members
        """

        if self.bounds == None:
            return population, [ (False,0) for i in \
                                range(self.population_size) ]

        violations = []
        valid_population = population.copy()
        for i, member in enumerate(population):
            valid_member, distance, validity = \
                self._nearest_valid_parameters(member, self.bounds)
            valid_population[i] = valid_member
            violations.append((validity, distance))

        return valid_population, violations

    def _boundary_correction(self, cost, violations):
        """
        Apply corrections to the cost of members that violated conditions

        This is just an ad hoc adjustment, will implement the following in the 
        future: Errata/Addenda for A Method for Handling
        Uncertainty in Evolutionary Optimization With an
        Application to Feedback Control of Combustion, Nikolaus Hansen, 2011
        """

        corrected_cost = []
        for i in range(self.population_size):
            if violations[i][0]:
                corrected_cost.append(cost[i] + 
                    violations[i][1] * np.median(cost) / (self.sigma * \
                        self.sigma * np.mean(covariance_matrix)))
            else:
                corrected_cost.append(cost[i])

        return corrected_cost

    def engage(self, objective_funct, args=(), iterations = 100, \
        parallel=True, num_of_jobs=-2, bounds=None, verbose=False):
        """
        Run the update process on the objective function for designated 
        number of iterations.
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

    def plot_cost_over_time(self, prefix='test', logy=True):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation,
        the mean, and the 25th/75th interquartile ranges.
        """

        import matplotlib.pyplot as plt

        sorted_cost_by_generation = [ [ member['cost'] \
            for member in generation ] \
            for generation in self.population_history]

        max_cost_by_generation = \
            np.min(sorted_cost_by_generation, axis=1)
        mean_cost_by_generation = \
            np.mean(sorted_cost_by_generation, axis=1)
        percentile25th_by_generation = \
            np.percentile(sorted_cost_by_generation, 25, axis=1)
        percentile75th_by_generation = \
            np.percentile(sorted_cost_by_generation, 75, axis=1)

        plt.errorbar(range(len(mean_cost_by_generation)), \
            mean_cost_by_generation, 
            yerr=[mean_cost_by_generation - percentile25th_by_generation, \
            percentile75th_by_generation - mean_cost_by_generation],
            marker='o', ls='-', color='blue', label='mean')
        plt.plot(range(len(max_cost_by_generation)), \
            max_cost_by_generation, ls='None', marker='x', \
            color='red', label='best')
        if logy:
            plt.yscale('log')
        plt.grid(True)
        plt.xlabel('generation')
        plt.ylabel('cost')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(prefix + "_evocost.png", dpi=300)
        plt.close()
        plt.clf()

def mutual_sort(sorting_sequence, *following_sequences, 
    reversed=False, key=None):

    sorted_indices = np.argsort(sorting_sequence)
    sorted_following_sequences = []
    for following_sequence in following_sequences:
        sorted_following_sequences.append(
            np.array(following_sequence)[sorted_indices])

    return (np.array(sorting_sequence)[sorted_indices], \
            *sorted_following_sequences)

def fmax(objective_funct, x0, sigma0, args=(), iterations=1000, \
    parallel=True, num_of_jobs=-2, cma_params={}, bounds=[], \
    verbose=False, return_history=False):
    """
    A functional version of the CMA evolutionary strategy

    bounds: a list of tuples in the order of x0. Tuple are the ranges of the 
    parameters accepted.
    """

    cma_object = CMAEvolutionaryStrategy(x0, sigma0, **cma_params)
    cma.engage(objective_funct, args, iterations, parallel, 
                num_of_jobs, bounds, verbose)

    if return_history:
        return self.all_time_best['x'], self.all_time_best['cost'], \
                self.population_history
    else:
        return self.all_time_best['x'], self.all_time_best['cost']

def elli(x):
    """ellipsoid-like test cost function"""
    n = len(x)
    aratio = 1e3
    return sum(x[i]**2 * aratio**(2.*i/(n-1)) for i in range(n))

def sphere(x):
    """sphere-like, ``sum(x**2)``, test cost function"""
    return sum(x[i]**2 for i in range(len(x)))

def rosenbrock(x):
    """Rosenbrock-like test cost function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater one')
    return sum(100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 
        for i in range(n-1))

class FunctorParallelTest:
    def __init__(self, par1, par2):
        self.par1 = None
        self.par2 = None
        
    def __call__(self, x, par1, par2):
        return self._elli(x, par1, par2)

    def _elli(self, x, par1, par2):
        """ellipsoid-like test cost function"""
        self.par1 = par1
        self.par2 = par2
        n = len(x)
        return sum(x[i]**2 * self.par1**(self.par2*i/(n-1)) for i in range(n))

if __name__ == '__main__':
    """
    testing
    """

    # cmaes = CMAEvolutionaryStrategy([0.5, 0.5, 0.5], 0.5)
    # cmaes.engage(rosenbrock, iterations=1000, num_of_jobs=3, verbose=True)
    # cmaes.plot_cost_over_time()

    cmaes = CMAEvolutionaryStrategy([0.5, 0.5, 0.5], 0.5)
    test_functor = FunctorParallelTest(1e3, 2.)
    cmaes.engage(test_functor, args=(1e3, 2.), iterations=1000, num_of_jobs=2, verbose=True)
    cmaes.plot_cost_over_time()