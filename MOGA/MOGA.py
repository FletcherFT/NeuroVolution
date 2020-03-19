import numpy as np
import random


class NDSAGA:
    def __init__(self, pop_size, elitism=0.1):
        self._n = pop_size
        self._e = elitism

    def _ndsa(self, solutions):
        # unpack the solutions
        idx, x, z = zip(*solutions)
        # For every solution, check if it dominates the rest
        for i in range(self._n):
            for j in range(i + 1, self._n):
                pass

    def update(self, fitness):
        pass


