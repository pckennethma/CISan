import numpy as np
import pandas as pd
import copy
from multiprocessing import Pool
from typing import List, Set, Dict
from functools import partial
from itertools import combinations
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag

MAX_ORDER = 5
ENABLE_PREFETCHING = True
REACHABLE = {}

from IndependenceSolver import KnowledgeBase
import IndependenceSolver
IndependenceSolver.CONSTRAINT_SLICING = ENABLE_PREFETCHING
from Utility import CIStatement
from GraphUtils import *
from DataUtils import read_table
from Kendall import DPKendalTau
from Chisq import Chisq

def pc_skl(var_num, independence_func, enable_solver=True): # Legacy code

    TOTAL_CI = 0
    
    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)
    
    for (node_x, node_y) in combinations(range(var_num), 2):
        TOTAL_CI += 1
        if independence_func(node_x, node_y, set()):
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
            kb.AddFact(CIStatement.createByXYZ(node_x, node_y, set(), True))
        else: 
            kb.AddFact(CIStatement.createByXYZ(node_x, node_y, set(), False))
    order = 1
    while order <= MAX_ORDER:
        edges_to_remove = set()
        for node_x in graph:
            neighbors = graph[node_x]
            for node_y in neighbors:
                if (node_y, node_x) in edges_to_remove or (node_x, node_y) in edges_to_remove: continue
                if len(neighbors) - 1 < order: continue
                print(node_x, node_y, edges_to_remove, graph, TOTAL_CI)
                cond_set_list = list(combinations(neighbors - {node_y}, order))
                ci_relation_candidate = [CIStatement.createByXYZ(node_x, node_y, set(cond_set), True) for cond_set in cond_set_list]
                if enable_solver:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        psan_outcome = kb.SinglePSan(ci)
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        if psan_outcome is None: 
                            is_ind = independence_func(x, y, z)
                            kb.AddFact(CIStatement.createByXYZ(x, y, z, is_ind))
                            print("CI Query", str(CIStatement.createByXYZ(x, y, z, is_ind)))
                        else: 
                            is_ind = psan_outcome.ci
                            assert is_ind == independence_func(x, y, z, True)
                            kb.AddFact(psan_outcome)
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        if independence_func(x, y, z):
                            edges_to_remove.add((node_x, node_y))
                            break
        for edge in edges_to_remove:
            node_x, node_y = edge
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
        order += 1

    return graph, TOTAL_CI


def PSan_pc_skl(var_num, independence_func, enable_solver=True):
    TOTAL_CI = 0

    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)

    # Done: add pruning check
    # Optional: optimize this for loop, and move it into the following while loop
    for order in range(MAX_ORDER + 1):
        edges_to_remove = set()
        for node_x in graph:
            neighbors = graph[node_x]
            for node_y in neighbors:
                if (node_y, node_x) in edges_to_remove or (node_x, node_y) in edges_to_remove:
                    continue
                if len(neighbors) - 1 < order:
                    continue
                print(node_x, node_y, edges_to_remove, graph, TOTAL_CI)
                cond_set_list = list(combinations(neighbors - {node_y}, order))
                ci_relation_candidate = [CIStatement.createByXYZ(
                    node_x, node_y, set(cond_set),
                    True) for cond_set in cond_set_list]
                if enable_solver:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        psan_outcome = kb.SinglePSan(ci)
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        if psan_outcome is None:
                            is_ind = independence_func(x, y, z)
                            kb.AddFact(CIStatement.createByXYZ(x, y, z, is_ind))
                            print("CI Query", str(CIStatement.createByXYZ(x, y, z, is_ind)))
                        else:
                            is_ind = psan_outcome.ci
                            assert is_ind == independence_func(x, y, z, True)
                            kb.AddFact(psan_outcome)
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
                        TOTAL_CI += 1
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        if independence_func(x, y, z):
                            edges_to_remove.add((node_x, node_y))
                            break
        for edge in edges_to_remove:
            node_x, node_y = edge
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)

    return graph, TOTAL_CI


def run_dpkt_pc(benchmark):
    dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
    data_path = f"data/{benchmark}-10k.csv"
    dag=read_dag(dag_path)
    dpkt = DPKendalTau(read_table(data_path), dag=dag)
    est, TOTAL_CI = pc_skl(dag.get_num_nodes(), dpkt.kendaltau_ci, True)
    return est, TOTAL_CI, dpkt.ci_invoke_count, dpkt.get_eps_prime()

def run_dpkt_pc_repeat(benchmark):
    with Pool() as pool:
        result = pool.map(run_dpkt_pc, [benchmark]*10)
    dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
    dag=read_dag(dag_path)
    avg_shd = np.mean([compare_skeleton(rlt[0], dag) for rlt in result])
    avg_total_ci = np.mean([rlt[1] for rlt in result])
    avg_ci_count = np.mean([rlt[2] for rlt in result])
    avg_eps = np.mean([rlt[3] for rlt in result])

    return avg_shd, avg_total_ci, avg_ci_count, avg_eps

def run_chisq_pc(benchmark):
    dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
    data_path = f"data/{benchmark}-10k.csv"
    dag=read_dag(dag_path)
    chisq = Chisq(read_table(data_path), dag=dag)
    est, TOTAL_CI = pc_skl(dag.get_num_nodes(), chisq.chisq_ci, True)
    return est, TOTAL_CI, chisq.ci_invoke_count

def run_oracle_pc(benchmark):
    dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
    dag=read_dag(dag_path)
    oracle = OracleCI(dag=dag)
    est, TOTAL_CI = pc_skl(dag.get_num_nodes(), oracle.oracle_ci, True)
    return est, TOTAL_CI, oracle.ci_invoke_count

if __name__ == "__main__":

    # it seems that Z3 have some bug on asia
    benchmarks = ["earthquake", "survey", "cancer", "sachs",]
    benchmarks = ["earthquake"]

    result = {
        bn: {"shd": [], "#CI Test": [], "#CI Query": [], "Eps": []} for bn in benchmarks
    }

    for benchmark in benchmarks:

        dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
        dag=read_dag(dag_path)
        # dag = read_dag(dag_path)
        # REACHABLE = {}
        # NUM_OF_CI_TEST = 0
        # TOTAL_CI = 0
        # print("reachable", REACHABLE)
        # print("NUM_OF_CI_TEST",NUM_OF_CI_TEST, "TOTAL_CI", TOTAL_CI)

        
    #     # kendal tau
    #     for rlt in results:
    #         est, TOTAL_CI, ci_invoke_count, eps_prime = rlt
    #         result[benchmark]["shd"].append(compare_skeleton(est, dag))
    #         result[benchmark]["#CI Test"].append(ci_invoke_count)
    #         result[benchmark]["#CI Query"].append(TOTAL_CI)
    #         result[benchmark]["Eps"].append(eps_prime)
        
    # # for benchmark in benchmarks:
    #     print(benchmark)
    #     print("SHD", np.mean(result[benchmark]["shd"]), np.std(result[benchmark]["shd"]))
    #     print("NUM_OF_CI_TEST", np.mean(result[benchmark]["#CI Test"]), np.std(result[benchmark]["#CI Test"]))
    #     print("TOTAL_CI", np.mean(result[benchmark]["#CI Query"]), np.std(result[benchmark]["#CI Query"]))
    #     print("Eps", np.mean(result[benchmark]["Eps"]), np.std(result[benchmark]["Eps"]))

        
        # est, TOTAL_CI, ci_invoke_count = run_oracle_pc(benchmark)
        # shd = compare_skeleton(est, dag)
        est, TOTAL_CI, ci_invoke_count = run_oracle_pc(benchmark)
        print(benchmark)
        # print("SHD", )
        print("SHD", compare_skeleton(est, dag))
        print("NUM_OF_CI_TEST", ci_invoke_count)
        print("TOTAL_CI", TOTAL_CI)
        # print("EPS", avg_eps)
        

