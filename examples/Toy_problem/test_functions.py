import tensorflow as tf
import math

"""
This file provides a set of test functions for assessing the methodology on more traditional optimization style problems
with no surrogate modelling of the stages, instead employing the models directly
"""

class DTLZ7a():

    def g(self, x):
        g = 1 + 9/6 * tf.reduce_sum(x[2:])
        return g

    def h(self, x, g):
        h = 3 - x[0]/(1+g)*( 1 + math.sin(3 * math.pi * x[0]) )
        h -= x[1]/(1+g)*( 1 + math.sin(3 * math.pi * x[1]) )