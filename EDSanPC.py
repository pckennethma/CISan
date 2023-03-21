import numpy as np
import pandas as pd
import json
from multiprocessing import Pool
from typing import List, Set, Dict
from functools import partial
from itertools import combinations
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
from datetime import datetime
import argparse

MAX_ORDER = 5
ENABLE_PREFETCHING = True
REACHABLE = {}

from IndependenceSolver import KnowledgeBase, FUNCTION_TIME_DICT
import IndependenceSolver
IndependenceSolver.CONSTRAINT_SLICING = ENABLE_PREFETCHING
from Utility import CIStatement
from GraphUtils import *
from DataUtils import read_table
from Kendall import DPKendalTau
from Chisq import Chisq

def EDsan_pc_skl(var_num, independence_func, enable_solver=True): 
    TOTAL_CI = 0
    graph = {i: set(range(var_num)) - {i} for i in range(var_num)}
    kb = KnowledgeBase([], var_num, False)
    
    for (node_x, node_y) in combinations(range(var_num), 2):
        TOTAL_CI += 1
        is_ind = independence_func(node_x, node_y, set())
        ci = CIStatement.createByXYZ(node_x, node_y, set(), is_ind)
        kb.EDSan(ci)
        kb.AddFact(ci)
        if is_ind:
            graph[node_x].remove(node_y)
            graph[node_y].remove(node_x)
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
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                        is_ind = independence_func(x, y, z)
                        incoming_ci = CIStatement.createByXYZ(x, y, z, is_ind)
                        kb.EDSan(incoming_ci)
                        kb.AddFact(incoming_ci)
                        print("CI Query", str(incoming_ci))
                        if is_ind:
                            edges_to_remove.add((node_x, node_y))
                            break
                else:
                    for ci in ci_relation_candidate:
                        x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
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


def run_error_injection_oracle_pc(benchmark, error_rate=0.1):
    dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
    dag=read_dag(dag_path)
    oracle = ErrorInjectionOracleCI(dag=dag, error_rate=error_rate)
    error_detected = False
    try:
        EDsan_pc_skl(dag.get_num_nodes(), oracle.oracle_ci, True)
    except AssertionError:
        error_detected = True
    return error_detected, oracle.error_num, oracle.ci_invoke_count, oracle.error_injection_position

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", "-b", type=str,
                        choices=["earthquake", "survey", "cancer", "sachs"],
                        default="earthquake")

    args = parser.parse_args()

    
    benchmark = args.benchmarks
    print("=========================================")

    result = {"error_rate": [], "error_detected": [], "error_count": [], "ci_count": [], "error_injection_position": []} 
        
    for error_rate in np.linspace(0.01, 0.1, 10):
        print("=========================================")
        print(f"Benchmark: {benchmark}, Error rate: {error_rate}")
        start_time = datetime.now()
        dag_path = f"/home/pmaab/ML4C/benchmarks/{benchmark}_graph.txt"
        dag=read_dag(dag_path)
        error_detected, error_count, ci_count, error_injection_position = run_error_injection_oracle_pc(benchmark, error_rate)
        end_time = datetime.now()
        print("Time taken: ", end_time - start_time)
        print("Error detected: ", error_detected)
        print("Error count: ", error_count)
        print("CI count: ", ci_count)
        print("Error injection position: ", error_injection_position)
        result["error_rate"].append(error_rate)
        result["error_detected"].append(error_detected)
        result["error_count"].append(error_count)
        result["ci_count"].append(ci_count)
        result["error_injection_position"].append(error_injection_position)
        for key, val in FUNCTION_TIME_DICT.items():
            print(f"Function: {key}, Total time: {val[0]}, Total count: {val[1]}")
        
    json.dump(result, open(f"result/{benchmark}_EDSanPC.json", "w"))    
