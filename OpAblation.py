from Chisq import Chisq
from Kendall import DPKendalTau
from DataUtils import read_table
from GraphUtils import *
from Utility import CIStatement
import IndependenceSolver
from IndependenceSolver import KnowledgeBase, FUNCTION_TIME_DICT
import numpy as np
import pandas as pd
import copy
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

IndependenceSolver.CONSTRAINT_SLICING = ENABLE_PREFETCHING


def EDsan_pc_skl(var_num, independence_func, enable_solver=False):
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
    return graph, TOTAL_CI, kb


def run_detection(
        kb, error_rate: float, seed: int, method_name: str):

    kb.Perturb(error_rate, seed)
    last_ci = kb.facts.pop()
    try:
        kb.EDsan_singleM(last_ci, method_name)
    except AssertionError:
        error_detected = True
        print("Error Detected")
        print("======================================")
    return error_detected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", type=str,
                        choices=["earthquake", "survey", "cancer", "sachs"],
                        default="sachs")
    parser.add_argument("--method-name", "-m", type=str,
                        default="Graphoid",
                        choices=["Graphoid", "Slicing"])
    parser.add_argument("--error-ratio", "-r", type=float, default=0.01)
    args = parser.parse_args()
    print(args)
    print("=========================================")

    start_time = datetime.now()
    detected_num = 0
    total_num = 100
    dag_path = f"/home/zjiae/Data/benchmarks/{args.benchmark}_graph.txt"
    dag = read_dag(dag_path)
    oracle = OracleCI(dag=dag)
    _, _, kb = EDsan_pc_skl(dag.get_num_nodes(), oracle.oracle_ci, enable_solver=False)
    for seed in range(total_num):
        if run_detection(kb, args.error_ratio, seed, args.method_name):
            detected_num += 1
    print(f"Detected {detected_num} out of {total_num} errors")

    end_time = datetime.now()
    last_time = end_time - start_time
    print("Time taken: ", last_time)
    for key, val in FUNCTION_TIME_DICT.items():
        print(f"Function: {key}, Total time: {val[0]}, Total count: {val[1]}")
