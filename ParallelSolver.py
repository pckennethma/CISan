from Rules import RuleContext
from Utility import *
from multiprocessing import Manager, Process, Pool
from typing import List, Tuple

INCONSISTENT_KB = "INCONSISTENT_KB"
class ParallelSlicingSolver:

    rule_set = ["symmetric_rule", "decomposition_rule", "weak_union_rule", 
                       "contraction_rule", "intersection_rule", "composition_rule",
                    #    "weak_transitivity_rule", 
                       "chordality_rule"]

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
    
    # def check_consistency(self, ):
    #     self.return_dict = self.manager.dict()
    #     jobs: List[Process] = []
    #     for idx, rule_name in enumerate(ParallelSlicingSolver.rule_set):
    #         p1 = Process(target=ParallelSlicingSolver.worker, args=(idx+1, rule_name, self.var_num, self.ci_facts + [self.incoming_ci], self.timeout, self.return_dict))
    #         jobs.append(p1)
    #         p1.start()
    #     for proc in jobs:
    #         proc.join()

    #     return not any(map(lambda x: x == unsat, self.return_dict.values()))
    

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

class ParallelHybridEDSanSolver:
    rule_set = ["symmetric_rule", "decomposition_rule", "weak_union_rule", 
                       "contraction_rule", "intersection_rule", "composition_rule",
                       "chordality_rule"]
    
    dump_unsat_core = False

    def __init__(self, var_num: int, ci_facts: List[CIStatement], incoming_ci: CIStatement, slicing_timeout:int, full_timeout:int):
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.slicing_timeout = slicing_timeout
        self.full_timeout = full_timeout
        self.manager= Manager()

    def check_consistency(self)->Tuple[bool, bool]:
        self.return_dict = self.manager.dict()
        jobs: List[Process] = []

        fp = Process(target=ParallelHybridEDSanSolver.worker, args=(-1, "full", self.var_num, self.ci_facts + [self.incoming_ci], self.full_timeout, self.return_dict))
        fp.start()

        for idx, rule_name in enumerate(ParallelHybridEDSanSolver.rule_set):
            p = Process(target=ParallelHybridEDSanSolver.worker, args=(idx, rule_name, self.var_num, self.ci_facts + [self.incoming_ci], self.slicing_timeout, self.return_dict))
            jobs.append(p)
            p.start()
        
        for proc in jobs:
            proc.join()
        if unsat in self.return_dict.values(): return False, True
        fp.join()
        return (unsat not in self.return_dict.values()), False

    @staticmethod
    def worker(index:int ,rule_name: str, var_num:int, ci_facts: List[CIStatement], timeout:int, return_dict):
        solver = SolverFor("QF_UFBV")
        if ParallelHybridEDSanSolver.dump_unsat_core:
            solver.set(unsat_core=True)
        ci_euf = Function("ci_euf", BitVecSort(var_num), BitVecSort(var_num), BitVecSort(var_num), BitVecSort(2))
        rule_ctx = RuleContext(var_num, ci_euf)
        if rule_name == "full":
            for rule in rule_ctx.constraints.items(): 
                if ParallelHybridEDSanSolver.dump_unsat_core:
                    solver.assert_and_track(rule[1], rule[0])
                else:
                    solver.add(rule[1])
        else:
            solver.add(rule_ctx.constraints["initial_validity_condition"])
            solver.add(rule_ctx.constraints[rule_name])
        for ci in ci_facts:
            if rule_name == "full" or ci_facts[-1].has_overlap(ci):
                if ParallelHybridEDSanSolver.dump_unsat_core and rule_name == "full":
                    solver.assert_and_track(ci.generate_constraint(ci_euf, var_num), str(ci))
                else: solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        return_dict[index] = solver.check()
        if ParallelHybridEDSanSolver.dump_unsat_core and solver.unsat_core() != []:
            print("Unsat core:", solver.unsat_core())

class ParallelGraphoidEDSanSolver:

    def __init__(self, var_num: int, ci_facts: List[CIStatement], incoming_ci: CIStatement):
        self.var_num = var_num
        self.new_facts = ci_facts.copy()
        self.new_facts.append(incoming_ci)
        self.variables = psitip.rv(*[f"X{i}" for i in range(self.var_num)])
        self.ci_statements = [fact.graphoid_expr(self.variables) for fact in self.new_facts if fact.ci]
        self.source_expr = psitip.alland(self.ci_statements)
        self.cd_terms = [fact.graphoid_term(self.variables) for fact in self.new_facts if not fact.ci]
        self.manager= Manager()

    def check_consistency(self):
        self.return_dict = self.manager.dict()
        pool = Pool(processes=48)

        for _, cd_term in enumerate(self.cd_terms):
            pool.apply_async(ParallelGraphoidEDSanSolver.worker, args=(cd_term, self.source_expr, self.return_dict))
        pool.close()
        pool.join()

        return True not in self.return_dict.values()

    @staticmethod
    def worker(cd_term: psitip.Expr, source_expr: psitip.Region, return_dict):
        return_dict[cd_term] = source_expr.get_bayesnet().check_ic(cd_term)

class ParallelPSanFullSolver:

    def __init__(self, var_num: int, ci_facts: List[CIStatement], incoming_ci: CIStatement, timeout:int, max_workers=32):
        self.max_workers = max_workers
        self.var_num = var_num
        self.ci_facts = ci_facts
        self.incoming_ci = incoming_ci
        self.timeout = timeout
        self.manager= Manager()

    def check_pruning(self):
        self.return_dict = self.manager.dict()

        p1 = Process(target=ParallelPSanFullSolver.worker, args=(self.incoming_ci, self.var_num, self.ci_facts, self.timeout, self.return_dict))
        p2 = Process(target=ParallelPSanFullSolver.worker, args=(self.incoming_ci.get_negation(), self.var_num, self.ci_facts, self.timeout, self.return_dict))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        if self.return_dict[str(self.incoming_ci)] == unsat and self.return_dict[str(self.incoming_ci.get_negation())] == unsat:
            return INCONSISTENT_KB
        if self.return_dict[str(self.incoming_ci)] == unsat:
            return self.incoming_ci.get_negation()
        if self.return_dict[str(self.incoming_ci.get_negation())] == unsat:
            return self.incoming_ci
        return None
    
    @staticmethod
    def worker(incoming_ci: CIStatement, var_num:int, ci_facts: List[CIStatement], timeout:int, return_dict):
        solver = SolverFor("QF_UFBV")
        ci_euf = Function("ci_euf", BitVecSort(var_num), BitVecSort(var_num), BitVecSort(var_num), BitVecSort(2))
        rule_ctx = RuleContext(var_num, ci_euf)
        for name, constraint in rule_ctx.constraints.items():
            solver.add(constraint)
        for ci in ci_facts:
            solver.add(ci.generate_constraint(ci_euf, var_num))
        solver.add(incoming_ci.generate_constraint(ci_euf, var_num))
        solver.set("timeout", timeout)
        return_dict[str(incoming_ci)] = solver.check()




