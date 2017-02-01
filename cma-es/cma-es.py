import numpy as np
import matplotlib.pyplot as plt

class CMAEvolutionaryStrategy:
	"""
	See below for implementation details:
	Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772. https://doi.org/10.1007/11007937_4
	"""

	def __init__(self, **kwargs):

		