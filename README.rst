Implements the CMA-ES algorithm of Hansen.
Has optional parallelization.
Works with Python2.7 and Python3.5

Consider re-scaling the parameters before using CMA-ES.
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
    will be roughly 2, see also below. The natural encoding of (some of) the 
    parameters can also be "logarithmic". That is, for a parameter that must 
    always be positive, with a ratio between typical upper and lower value 
    being larger than 100, we might use 10x instead of x to call the 
    objective function. More specifically, to achieve the parameter range 
    [10-4,10-1], we use 10-4x103x/10 with x in [0; 10]. Again, the idea is to 
    have similar sensitivity: this makes sense if we expect the change from 
    10-4 to 10-3 to have an impact similar to the change from 10-2 to 10-1. 
    In order to avoid the problem that changes of very small values have too 
    less an impact, an alternative is to choose 10-1 x (x/10)2 >= 0. In the 
    case where only a lower bound at zero is necessary, a simple and natural 
    transformation is x2 × default_x, such that x=1 represents the default 
    (or initial) value and x remains unbounded during optimization.

    In summary, to map the values [0;10] into [a;b] we have the alternative 
    transformations :
    a + (b-a) x x/10 or a + (b-a) x (x/10)2 >= a or a x (b/a)x/10 >= 0.


Requirements:
(1) Numpy
(2) Matplotlib
(3) joblib
(4) utilities (https://github.com/Nathaniel-Rodriguez/utilities.git)