import math
import numpy as np
import pandas  as pd
from itertools import combinations
from scipy.spatial import distance_matrix


def euc_distance(a, b):
    d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return d


def prob_set_distance(o, k, approximation):
    neighbours = np.argsort(o)[1:k + 1]
    S = o[neighbours]
    standard_distance = math.sqrt(sum(S ** 2) / k)
    pdist = approximation * standard_distance
    return pdist, neighbours


class LoOP:
    def __init__(self, data, k, approximation):
        self.data = data
        self.k = k
        self.approximation = approximation
        self.size = len(self.data)
        self.dist_matrix = np.zeros((self.size, self.size), dtype=float)
        self.pdist_all = []
        self.plof = []
        self.nplof = None
        self.LoOP = []

    def distance_matrix_iter(self):     # not used, took too long execute compared to scipy + computes 2 arrays only
        comb = combinations(range(self.size), 2)
        counter = 0
        for i, j in list(comb):
            print(counter)
            d = euc_distance(self.data[i], self.data[j])
            self.dist_matrix[i, j] = d
            self.dist_matrix[j, i] = d
            counter += 1

    def distance_matrix(self):     # other types of distnce calculations could be used
        self.dist_matrix = distance_matrix(self.data, self.data)

    def prob_local_outlier_factor(self):
        for i in self.dist_matrix:
            pdist, neighbours = prob_set_distance(i, self.k, self.approximation)
            self.pdist_all.append([pdist, neighbours])

        for pdist, neighbours in self.pdist_all:
            E = 0
            for neighbour_k in neighbours:
                pdist_k = self.pdist_all[neighbour_k][0]
                E += pdist_k
            self.plof.append(pdist / (E / self.k) - 1)

    def norm_prob_local_outlier_factor(self):
        eplof = np.mean([i ** 2 for i in self.plof])
        self.nplof = self.approximation * math.sqrt(eplof)

    def local_outlier_prob(self):
        factor = self.nplof * math.sqrt(2)
        for i in self.plof:
            loop_i = max(0.0, math.erf(i / factor))
            self.LoOP.append(loop_i)

    def fit(self):
        self.distance_matrix()
        self.prob_local_outlier_factor()
        self.norm_prob_local_outlier_factor()
        self.local_outlier_prob()
        return self.LoOP
