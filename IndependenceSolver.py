import random
from typing import List
from p_tqdm import p_umap
from functools import partial
from itertools import chain
from multiprocessing import Pool
from Rules import RuleContext
from Utility import *
from ParallelSolver import ParallelSlicingSolver, ParallelPSanFullSolver, INCONSISTENT_KB

CONSTRAINT_SLICING = True
MAX_BACKTRACK_THRESHOLD = 10
ENABLE_PARALLEL = True
ENABLE_GRAPHOID = True
ENABLE_MARGINAL_OMITTING = True # if true, we will omit Psan and EDsan if the pool only contains marginal statements
# psitip.PsiOpts.setting(solver = "pyomo.glpk")

class KnowledgeBase:
    def __init__(self, global_facts: List[CIStatement], var_num:int, do_trace=True):
        self.facts = global_facts
        self.var_num = var_num
        self.do_track = do_trace
        self.backtrack_count = 0
    
    def AddFact(self, fact: CIStatement):
        self.facts.append(fact)
    
    def AddFacts(self, facts: List[CIStatement]):
        self.facts += facts

    # def ConstructKB(self):
    #     converted_facts = []
    #     for ind in self.facts:
    #         x, y, z, is_ind = ind
    #         converted_facts.append(
    #             (
    #                 KnowledgeBase.GenerateBitVal(x, self.var_num),
    #                 KnowledgeBase.GenerateBitVal(y, self.var_num),
    #                 KnowledgeBase.GenerateBitVal(z, self.var_num),
    #                 is_ind
    #             )
    #         )
    #     return converted_facts
    
    # @staticmethod
    # def Involves(raw_fact, node):
    #     return node in raw_fact[0] or node in raw_fact[1] or node in raw_fact[2]
    
    # @staticmethod
    # def StronglyInvolves(raw_fact : List[set], local_nodes: set):
    #     return local_nodes.intersection(raw_fact[0]) and local_nodes.intersection(raw_fact[1])        
    
    # def ConstructMiniKB(self, local_nodes: set):
    #     local_facts = []
    #     for raw_fact in self.raw_facts:
    #         for node in local_nodes:
    #             if KnowledgeBase.Involves(raw_fact, node):
    #                 local_facts.append(raw_fact)
    #                 break
    #     return self.ConstructKB(local_facts)
    
    @staticmethod
    def GenerateConstraints(facts: List[CIStatement], ci_euf: FuncDeclRef, var_num:int):
        constraints = []
        for fact in facts:
            constraints.append(
                fact.generate_constraint(ci_euf, var_num)
            )
        return And(constraints)
    
    @staticmethod
    def GenerateLightWeightConstraints(facts: List[CIStatement], target:CIStatement, ci_euf: FuncDeclRef, var_num:int):
        constraints = []
        for fact in facts:
            if fact.has_overlap(target):
                constraints.append(
                    fact.generate_constraint(ci_euf, var_num)
                )
        return And(constraints)

    # the first return value denotes whether we obtain meaningful result
    # the second return value denotes whether the hyp holds
    def PSanFullLegacy(self, hyp: CIStatement, prune_neg=False):
        # return: [int,bool] first bool variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1); second bool variable denotes
        # whether the solver returns unknown
        
        # if local_nodes != None:
        #     facts, var_map = self.ConstructMiniKB(local_nodes)
        # else:
        # facts, var_map = self.ConstructKB()

        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(ci_euf, self.var_num)

        kb_constraint = KnowledgeBase.GenerateConstraints(self.facts, ci_euf, self.var_num)

        rule_ctx = RuleContext(self.var_num, ci_euf)
        
        timeout = self.compute_timeout("psan_full")

        # pos_solver = SolverFor("QF_UFBV")
        pos_solver = Solver()
        pos_solver.set("timeout", timeout)
        # neg_solver = SolverFor("QF_UFBV")
        neg_solver = Solver()
        neg_solver.set("timeout", timeout)

        if not self.do_track:
            pos_solver.add(hyp_constraint)
            neg_solver.add(neg_hyp_constraint)
            pos_solver.add(kb_constraint)
            neg_solver.add(kb_constraint)
        else:
            pos_solver.assert_and_track(hyp_constraint, "hyp")
            print(pos_solver.check(), "hyp_constraint")
            neg_solver.assert_and_track(neg_hyp_constraint, "not hyp")
            pos_solver.assert_and_track(kb_constraint, "kb")
            print(pos_solver.check(), "kb_constraint")
            neg_solver.assert_and_track(kb_constraint, "kb")

        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            if not self.do_track:
                pos_solver.add(constraint)
                neg_solver.add(constraint)
            else:
                pos_solver.assert_and_track(constraint, name)
                neg_solver.assert_and_track(constraint, name)
        pos_rlt = pos_solver.check()
        if pos_rlt == unsat and prune_neg:
            return 0, True
        neg_rlt = neg_solver.check()
        if pos_rlt == unsat and neg_rlt == sat:
            return 0, False
        elif pos_rlt == sat and neg_rlt == unsat:
            return 1, False
        elif pos_rlt == unsat and neg_rlt == unknown:
            return 0, True
        elif pos_rlt == unknown and neg_rlt == unsat:
            return 1, True
        elif pos_rlt == unknown and neg_rlt == unknown:
            return -1, True
        elif pos_rlt == sat and neg_rlt == sat:
            return -1, False
        elif pos_rlt == sat and neg_rlt == unknown:
            return -1, True
        elif pos_rlt == sat and neg_rlt == unknown:
            return -1, True
        else:
            return -2, False
    
    # the first return value denotes whether we obtain meaningful result
    # the second return value denotes whether the hyp holds
    def PSanSlicing(self, hyp: CIStatement, prune_neg=False):
        # return: [int] first variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1)

        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(ci_euf, self.var_num)

        kb_constraint = KnowledgeBase.GenerateLightWeightConstraints(self.facts, hyp, ci_euf, self.var_num)

        rule_ctx = RuleContext(self.var_num, ci_euf)
        
        timeout = self.compute_timeout("psan_slicing")

        pos_solvers = {i:SolverFor("QF_UFBV") for i in rule_ctx.constraints}
        neg_solvers = {i:SolverFor("QF_UFBV") for i in rule_ctx.constraints}

        indistinguishable = [sat, unknown]

        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            pos_solvers[name].set("timeout", timeout)
            neg_solvers[name].set("timeout", timeout)
            pos_solvers[name].add(constraint)
            neg_solvers[name].add(constraint)
            pos_solvers[name].add(hyp_constraint)
            neg_solvers[name].add(neg_hyp_constraint)
            pos_solvers[name].add(kb_constraint)
            neg_solvers[name].add(kb_constraint)
            pos_rlt = pos_solvers[name].check()
            if pos_rlt == unsat and prune_neg:
                return 0, True
            neg_rlt = neg_solvers[name].check()
            if pos_rlt in indistinguishable and neg_rlt in indistinguishable: continue
            if pos_rlt == unsat: return 0
            if neg_rlt == unsat: return 1
        return -1

    def PSanFullLegacy(self, hyp: CIStatement):
        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))

        hyp_constraint = hyp.generate_constraint(ci_euf, self.var_num)
        neg_hyp_constraint = hyp.get_negation().generate_constraint(ci_euf, self.var_num)

        kb_constraint = KnowledgeBase.GenerateConstraints(self.facts, ci_euf, self.var_num)

        rule_ctx = RuleContext(self.var_num, ci_euf)
        
        timeout = self.compute_timeout("psan_full")

        # pos_solver = SolverFor("QF_UFBV")
        pos_solver = Solver()
        pos_solver.set("timeout", timeout)
        # neg_solver = SolverFor("QF_UFBV")
        neg_solver = Solver()
        neg_solver.set("timeout", timeout)
        pos_solver.add(hyp_constraint)
        neg_solver.add(neg_hyp_constraint)
        pos_solver.add(kb_constraint)
        neg_solver.add(kb_constraint)
        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            pos_solver.add(constraint)
            neg_solver.add(constraint)
        pos_rlt = pos_solver.check()
        if pos_rlt == unsat:
            return hyp.get_negation()
        neg_rlt = neg_solver.check()
        if neg_rlt == unsat:
            return hyp
        return None

    def PSanFullParallel(self, hyp: CIStatement):
        ps = ParallelPSanFullSolver(self.var_num, self.facts, hyp, self.compute_timeout("psan_full"))
        return ps.check_pruning()

    def PSanSlicingParallel(self, hyp: CIStatement):
        # return: [int] first variable denotes whether hyp is true (1)
        # or false (0) or non-deterministic (-1)
        ps = ParallelSlicingSolver(self.var_num, [fact for fact in self.facts if fact.has_overlap(hyp)], hyp, self.compute_timeout("psan_slicing"))
        return ps.check_pruning()
        # if rlt is None:
        #     return -1
        # elif rlt.ci == hyp.ci: 
        #     return 1
        # else:
        #     return 0

    def Graphoid(self, hyp: CIStatement):
        new_facts = self.facts.copy()
        new_facts.append(hyp)
        return self.graphoid_consistency_checking(new_facts)
    
    def CheckConsistency(self): # Todo: reimplement in accordance with the EDSan
        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))
        kb_constraint = KnowledgeBase.GenerateConstraints(self.facts, ci_euf, self.var_num)
        solver = SolverFor("QF_UFBV")
        timeout = int(max(10_000, 1000 * self.var_num * 2))
        solver.set("timeout", timeout)
        solver.add(kb_constraint)
        rule_ctx = RuleContext(self.var_num, ci_euf)
        for rule in rule_ctx.constraints.items(): 
            solver.add(rule[1])
        check_rlt = solver.check()
        return check_rlt == sat or check_rlt == unknown

    def EDSanFull(self, incoming_ci: CIStatement):
        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))
        kb_constraint = KnowledgeBase.GenerateConstraints(self.facts, ci_euf, self.var_num)
        incoming_constraint = incoming_ci.generate_constraint(ci_euf, self.var_num)
        solver = SolverFor("QF_UFBV")
        timeout = self.compute_timeout("edsan_full")
        solver.set("timeout", timeout)
        solver.add(kb_constraint)
        solver.add(incoming_constraint)
        rule_ctx = RuleContext(self.var_num, ci_euf)
        for rule in rule_ctx.constraints.items(): 
            solver.add(rule[1])
        check_rlt = solver.check()
        return check_rlt == sat or check_rlt == unknown

    def EDSanSlicing(self, incoming_ci: CIStatement):
        ci_euf = Function("ci_euf", BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(self.var_num), BitVecSort(2))
        kb_constraint = KnowledgeBase.GenerateLightWeightConstraints(self.facts, incoming_ci, ci_euf, self.var_num)
        incoming_constraint = incoming_ci.generate_constraint(ci_euf, self.var_num)
        rule_ctx = RuleContext(self.var_num, ci_euf)
        solvers = {i:SolverFor("QF_UFBV") for i in rule_ctx.constraints}
        timeout = self.compute_timeout("esdan_slicing")
        for rule in rule_ctx.constraints.items():
            name, constraint = rule
            solvers[name].add(constraint)
            solvers[name].add(incoming_constraint)
            solvers[name].add(kb_constraint)
            solvers[name].set("timeout", timeout)
            rlt = solvers[name].check()
            if rlt == unsat: return False
        return True

    def EDSan(self, incoming_ci: CIStatement): # Todo: implement another PC.py for EDSan
        if ENABLE_MARGINAL_OMITTING:
            pass # Todo: implement marginal omittings
        if ENABLE_GRAPHOID:
            assert self.Graphoid(incoming_ci) == True
        if CONSTRAINT_SLICING:
            assert self.EDSanSlicing(incoming_ci)
        assert self.EDSanFull(incoming_ci)
    
    def Backtracking(self):
        print("start backtracking KB")
        self.backtrack_count += 1
        while not self.CheckConsistency():
            dropped = self.facts.pop()
            print("drop", str(dropped))
            

    def BatchPSan(self, hyps: List[CIStatement]): # Legacy code
        confirmed_ci = []
        if ENABLE_GRAPHOID:
            for hyp in hyps:
                if self.Graphoid(hyp) == False:
                    confirmed_ci.append(hyp.get_negation())
                    hyps.remove(hyp)
                elif self.Graphoid(hyp.get_negation()) == False:
                    confirmed_ci.append(hyp)
                    hyps.remove(hyp)
        
        # lightweight checking
        if CONSTRAINT_SLICING:
            if ENABLE_PARALLEL:
                with Pool() as pool:
                    check_results = pool.map(partial(self.PSanSlicing), hyps)
            else:
                check_results = list(map(partial(self.PSanSlicing), hyps))
            results = {hyps[idx]: result for idx, result in enumerate(check_results)}
            confirmed_ci = [hyps[idx] if result == 1 else hyps[idx].get_negation() for idx, result in enumerate(check_results) if result != -1]
            remained_hyps = [hyp for  hyp in hyps if results[hyp] == -1]
        else:
            remained_hyps = hyps
        # complete checking
        if ENABLE_PARALLEL:
            with Pool() as pool:
                check_results = pool.map(partial(self.PSanFullLegacy), remained_hyps)
        else:
            check_results = list(map(partial(self.PSanFullLegacy), remained_hyps))
        results = {remained_hyps[idx]: result for idx, result in enumerate(check_results)}
        if len([r for r in results if results[r][0] == -2]) != 0: 
            self.Backtracking()
            return []
        for ci in results:
            ci:CIStatement
            status, _ = results[ci]
            if status == 0: confirmed_ci.append(ci.get_negation())
            elif status == 1: confirmed_ci.append(ci)
        return confirmed_ci

    def SinglePSan(self, hyp: CIStatement): # Todo: implemention of PSan
        if ENABLE_MARGINAL_OMITTING:
            pass # Todo: implement marginal omitting
        if ENABLE_GRAPHOID:
            graphoid_outcome = self.graphoid_pruning(hyp)
            if graphoid_outcome is not None:
                print("graphoid:", graphoid_outcome, "is inferred")
                return graphoid_outcome
        if CONSTRAINT_SLICING:
            # psanslicing_result = self.PSanSlicing(hyp)
            psanslicing_outcome = self.PSanSlicingParallel(hyp)
            # if psanslicing_result == 0: 
            #     print("slicing:", hyp.get_negation(), "is inferred")
            #     return hyp.get_negation()
            # elif psanslicing_result == 1: 
            #     print("slicing:", hyp, "is inferred")
            #     return hyp
            if psanslicing_outcome is not None:
                print("slicing:", psanslicing_outcome, "is inferred")
                return psanslicing_outcome
        # psanfull_result = self.PSanFullLegacy(hyp)
        # if psanfull_result == 0: 
        #     print("full:", hyp.get_negation(), "is inferred")
        #     return hyp.get_negation()
        # elif psanfull_result == 1: 
        #     print("full:", hyp, "is inferred")
        #     return hyp
        psanfull_outcome = self.PSanFullParallel(hyp)
        if psanfull_outcome == INCONSISTENT_KB:
            self.Backtracking()
            return None
        if psanfull_outcome is not None:
            print("full:", psanfull_outcome, "is inferred")
            return psanfull_outcome
        print("full:", hyp, "is not inferred")
        return None

    def RecursivePSan(self, hyps: List[CIStatement], ind_func, step_size:int=1, early_stop:bool=False) -> List[CIStatement]: # Legacy code
        if self.backtrack_count >= MAX_BACKTRACK_THRESHOLD:
            print("max backtrack count reached, fallback to CI tests")
            return []
        result: List[CIStatement] = []
        sigma_star = hyps
        def _stop():
            if early_stop: 
                for ci in result: 
                    if ci.ci: return True
            return False

        while len(sigma_star) > 0:
            confirmed_ci = self.BatchPSan(sigma_star)
            self.AddFacts(confirmed_ci)
            result += confirmed_ci
            if _stop(): return result
            new_sigma_star = []
            for hyp in sigma_star:
                iso_ci = [i for i in confirmed_ci if hyp.is_isomorphic(i)]
                if len(iso_ci) == 0: new_sigma_star.append(hyp)
            sigma_star = new_sigma_star
            if len(sigma_star) == 0: break
            random.shuffle(sigma_star)
            for i in range(min(len(sigma_star), step_size)):
                ci:CIStatement = sigma_star.pop(0)
                assert len(ci.x) == 1 and len(ci.y) == 1
                x, y, z = list(ci.x)[0], list(ci.y)[0], ci.z
                new_ci = CIStatement.createByXYZ(x, y, z, ind_func(x, y, z))
                self.AddFact(new_ci)
                result.append(new_ci)
                if _stop(): return result
            
        return result

    
    def graphoid_consistency_checking(self, facts: List[CIStatement]) -> bool:
        variables = psitip.rv(*[f"X{i}" for i in range(self.var_num)])
        ci_statements = [fact.graphoid_expr(variables) for fact in facts if fact.ci]
        ci_facts = [fact for fact in facts if fact.ci]
        source_expr = psitip.alland(ci_statements)
        cd_term = [fact.graphoid_term(variables) for fact in facts if not fact.ci]
        cd_facts = [fact for fact in facts if not fact.ci]
        for idx, cd_term in enumerate(cd_term):
            # graphoid does not support disprove conditional independence
            # if target term is always independent given source statements, then there is an inconsistency in the graphoid
            if source_expr.get_bayesnet().check_ic(cd_term):
                print("input:", "\t".join([str(fact) for fact in ci_facts]))
                print("target:", cd_facts[idx])
                return False
        return True

    def graphoid_pruning(self, incoming_ci: CIStatement) -> bool:
        if not self.graphoid_consistency_checking(self.facts + [incoming_ci]):
            return incoming_ci.get_negation()
        elif not self.graphoid_consistency_checking(self.facts + [incoming_ci.get_negation()]):
            return incoming_ci
        else:
            return None
        # return source_expr.get_bayesnet().check_ic(target_term)

    def compute_timeout(self, check_type:str) -> int:
        if check_type == "psan_full":
            return int(max(30_000, 1000 * self.var_num * 2))
        elif check_type == "psan_slicing":
            return int(max(20_000, 1000 * len(self.facts)))
        elif check_type == "edsan_full":
            return int(max(30_000, 1000 * self.var_num * 2))
        elif check_type == "edsan_slicing":
            return int(max(20_000, 1000 * self.var_num * 1.5))
    


if __name__ == "__main__":

    # set_param("parallel.enable", True)
    # set_param("euf", True)

    # facts = [
    #     ({0}, {1}, {}, False),
    #     ({0}, {2}, {}, False),
    #     ({1}, {2}, {}, True),
    #     ({1}, {2}, {}, True),
    #     # ({1}, {2}, {3}, True),
    #     # ({2}, {3}, {1}, True),
    #     # ({1}, {2}, {2}, False)
    # ]

    # hyp = [({0}, {1}, {2}, True), ({3}, {1}, {2}, True), ({0}, {3}, {2}, True)]

    # kb = KnowledgeBase(list(map(CIStatement.create, facts)), var_num=4)

    # # is_hyp, has_unknown = kb.VerifyCIHypothesis(CIRelation.create(hyp))
    # def _mock_ind(x,y,z):
    #     print(x,y,z)
    #     return True
    # rlt = kb.RecursivePSan(list(map(CIStatement.create, hyp)), _mock_ind)
    # for ci in rlt:
    #     print(ci)
    # psitip.PsiOpts.setting(lptype = "H")
    X0, X1, X2, X3, X4= psitip.rv("X", "Y", "Z", "W", "U")
    facts = [
        psitip.I(X0&X1) == 0,
        psitip.I(X0&X2) < 0,
        psitip.I(X0&X3) < 0,
        psitip.I(X0&X4) < 0,
        psitip.I(X1&X2) < 0,
        psitip.I(X1&X3) < 0,
        psitip.I(X1&X4) < 0,
        psitip.I(X2&X3) < 0,
        psitip.I(X2&X4) < 0,
        psitip.I(X3&X4) < 0,
        psitip.I(X0&X2|X3) < 0,
        # psitip.I(X0&X2|X4) < 0,
        # psitip.I(X0&X3|X2) == 0,
        # psitip.I(X0&X4|X2) == 0,
        # psitip.I(X1&X2|X3) < 0,
        # psitip.I(X1&X2|X4) < 0,
        # psitip.I(X1&X3|X2) == 0,
        # psitip.I(X1&X4|X2) == 0,
        # psitip.I(X2&X0|X1) < 0,
    ]
    source_expr = psitip.alland(facts)
    print(source_expr.implies(psitip.I(X0&X2|X3) < 0))
