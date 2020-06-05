import numpy as np
from matplotlib import pyplot as plt
import math


class CMAC:
    def __init__(self, gen_factor, num_weights):
        self.gen_factor = gen_factor
        self.num_weights = num_weights
        self.weight_vec = np.ones(self.num_weights)
        self.association_map = {}

    def set_gen_factor(self, gen_factor):
        self.gen_factor = gen_factor