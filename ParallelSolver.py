from Rules import RuleContext
from Utility import *
from multiprocessing import Manager, Process
from typing import List

class ParallelSlicingSolver:

    rule_set = ["symmetric_rule", "decomposition_rule", "weak_union_rule", 
                       "contraction_rule", "intersection_rule", "composition_rule",
                       "weak_transitivity_rule", "chordality_rule"]

    def __init__(self, var_num: int, ci_facts: List[CIStatement], incoming_ci: CIStatement, timeout:int, max_workers=32):
        self.max_workers = max_workers
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.timeout = timeout
        self.manager= Manager()

    def check_consistency(self):
        self.return_dict = self.manager.dict()
        jobs: List[Process] = []
        for idx, rule_name in enumerate(ParallelSlicingSolver.rule_set):
            p = Process(target=ParallelSlicingSolver.worker, args=(idx, rule_name, self.var_num, self.ci_facts + [self.incoming_ci], self.timeout, self.return_dict))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        return unsat not in self.return_dict.values()

    def check_pruning(self):
        self.return_dict = self.manager.dict()
        jobs: List[Process] = []
        for idx, rule_name in enumerate(ParallelSlicingSolver.rule_set):
            p1 = Process(target=ParallelSlicingSolver.worker, args=(idx+1, rule_name, self.var_num, self.ci_facts + [self.incoming_ci], self.timeout, self.return_dict))
            p2 = Process(target=ParallelSlicingSolver.worker, args=(-idx-1, rule_name, self.var_num, self.ci_facts + [self.incoming_ci.get_negation()], self.timeout, self.return_dict))
            jobs.append(p1)
            jobs.append(p2)
            p1.start()
            p2.start()
        for proc in jobs:
            proc.join()
        for key in self.return_dict.keys():
            if key > 0:
                if self.return_dict[key] == unsat:
                    print("Pruning by rule:", ParallelSlicingSolver.rule_set[key-1])
                    return self.incoming_ci.get_negation()
            else:
                if self.return_dict[key] == unsat:
                    print("Pruning by rule:", ParallelSlicingSolver.rule_set[-key-1])
                    return self.incoming_ci
        return None

    @staticmethod
    def worker(index:int ,rule_name: str, var_num:int, ci_facts: List[CIStatement], timeout:int, return_dict):
        solver = SolverFor("QF_UFBV")
        ci_euf = Function("ci_euf", BitVecSort(var_num), BitVecSort(var_num), BitVecSort(var_num), BitVecSort(2))
        rule_ctx = RuleContext(var_num, ci_euf)
        solver.add(rule_ctx.constraints["initial_validity_condition"])
        solver.add(rule_ctx.constraints[rule_name])
        for ci in ci_facts:
            solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        return_dict[index] = solver.check()
