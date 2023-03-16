import numpy as np
from collections import namedtuple
from copy import copy
import warnings
from numpy import ma
from scipy.stats import mstats_basic
from scipy.stats._stats import _kendall_dis
import scipy.special as special
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from causallearn.graph.Dag import Dag
from causallearn.utils.cit import CIT

from GraphUtils import reachable
from Utility import CIStatement

class Chisq:
    def __init__(self, data: np.ndarray, alpha:float=0.01, small_dataset=False, dag:Dag=None):

        if small_dataset:
            self.data = data[:500]
        else:
            self.data = data
        self.alpha = alpha

        self.n = data.shape[0]
        self.node_size = data.shape[1]

        self.dag = dag
        self.oracle_cache = {}

        self.cit = CIT(self.data, "chisq")

        self.ci_invoke_count = 0

        self.accuracy = [0, 0]

    def chisq_ci(self, x: int, y: int, z: set[int], debug=False):
        # self.ci_invoke_count += 1
        if not debug: self.ci_invoke_count += 1
        rlt = self.cit(x, y, list(z)) > self.alpha
        self.independence_oracle(x, y, z, rlt)
        return rlt

    def independence_oracle(self, x: int, y: int, z: set[int], rlt):
        f_z = frozenset(z)
        x_reachable:dict = self.oracle_cache.setdefault(x, {})
        if f_z in x_reachable: x_z_reachable = x_reachable[f_z]
        else: x_z_reachable:set = frozenset(reachable(x, z, self.dag))
        true_rlt = y not in x_z_reachable
        if true_rlt == rlt:
            self.accuracy[0] += 1
        self.accuracy[1] += 1

        print("Tested by Oracle:", CIStatement.createByXYZ(x,y,z, true_rlt))
        print("Tested by Chisq:", CIStatement.createByXYZ(x,y,z, rlt))    