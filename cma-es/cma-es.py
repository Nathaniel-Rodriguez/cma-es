import math
import numpy as np
import matplotlib.pyplot as plt

class CMAEvolutionaryStrategy:
	"""
	Based on code and algorithm from Hansen, see below for implementation details:
	Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. https://doi.org/10.1007/11007937_4
	"""

	def __init__(self, x_0, **kwargs):
		"""
		x_0 : the initial centroid vector

		The problem variables should have been scaled, such that a single 
		standard deviation on all variables is useful and the optimum is 
		expected to lie within about x0 +- 3*sigma0.

		Can pass an array scaling_of_variables the designates the scale for
		each sigma for each variable independently, else assumed they are
		all the same scale for sigma.
		"""

		self.x_0 = np.array(x_0)
		self.num_of_dimensions = len(x_0)
		self.population_size = 4 + math.floor(3 * math.log(self.num_of_dimensions))
		self.num_of_parents = self.population_size / 2.
		self.weights = 
		parameter_defaults = {'sigma': 0.5,
			'scaling_of_variables': np.ones(self.num_of_dimensions)}

		for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

		self.effective_sigma = self.sigma * self.scaling_of_variables