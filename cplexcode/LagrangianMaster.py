import cplex
from cplex.exceptions import CplexSolverError
import sys
import copy
import numpy as np
import time

class LagrangianMaster:
    def __init__(self, network_scenario, sub_problem_results, num_time_slots, iter_counter, mu_ub):
        self.network_scenario = network_scenario
        self.num_time_slots = num_time_slots
        self.sub_problem_results = copy.deepcopy(sub_problem_results)
        self.iter_counter = iter_counter
        self.mu_ub = mu_ub

    def solve_master(self):
        # Add variables
        model = cplex.Cplex()
        model.variables.add(obj=[1 for t in range(self.num_time_slots)],
                            lb=[-cplex.infinity] * self.num_time_slots,
                            ub=[cplex.infinity] * self.num_time_slots,
                            types=["C"] * self.num_time_slots,
                            names=["theta_{}".format(t) for t in range(self.num_time_slots)])  # theta_t

        for s in range(self.network_scenario.num_apps):
            for j in range(self.network_scenario.num_DCs):
                model.variables.add(obj=[0] * self.num_time_slots,
                                    lb=[0] * self.num_time_slots,
                                    # ub=[cplex.infinity] * self.num_time_slots,
                                    ub=[self.mu_ub] * self.num_time_slots,
                                    types=["C"] * self.num_time_slots,
                                    names=["mu_{}_{}_{}".format(s,j,t) for t in range(self.num_time_slots)])  # mu_sjt

        # Set theta indices for use in the constraints
        theta = [*range(0, self.num_time_slots)]

        # Set mu indices
        mu = []
        index = self.num_time_slots
        for s in range(self.network_scenario.num_apps):
            mu.append([])
            for j in range(self.network_scenario.num_DCs):
                mu[s].append([])
                for t in range(self.num_time_slots):
                    mu[s][j].append(index)
                    index += 1

        # Add constraints
        time_index = self.network_scenario.start_time_index
        for t in range(self.num_time_slots):
            time_slot = time_index + t

            for counter in range(0, self.iter_counter):
                z_jt = self.sub_problem_results[counter][time_slot][3][0]
                w_isjt = self.sub_problem_results[counter][time_slot][2][0]
                gamma_st = self.sub_problem_results[counter][time_slot][6][0]
                x_sjt = self.sub_problem_results[counter][time_slot][4][0]
                on_sjt = self.sub_problem_results[counter][time_slot][5][0]

                index = [theta[t]]
                value = [1.0]

                rhs_value = 0
                # fixed / opening cost
                rhs_value += np.sum(np.multiply(z_jt, self.network_scenario.fixed_cost))

                # operational cost
                util = [0 for _ in range(self.network_scenario.num_DCs)]

                for j in range(self.network_scenario.num_DCs):
                    util[j] = 0
                    for s in range(self.network_scenario.num_apps):
                        for i in range(self.network_scenario.num_BSs):
                            util[j] += self.network_scenario.compute_factor[s] * \
                                       self.network_scenario.demand[i][s][time_slot] * \
                                       w_isjt[i][s][j+1]
                    util[j] = util[j] / self.network_scenario.compute_capacities[j]
                rhs_value += np.sum(np.multiply(util, self.network_scenario.op_cost))

                # cloud cost
                for i in range(self.network_scenario.num_BSs):
                    for s in range(self.network_scenario.num_apps):
                        rhs_value += (w_isjt[i][s][0]*self.network_scenario.cloud_cost[s])

                # Slack cost
                rhs_value += np.sum(np.multiply(gamma_st, self.network_scenario.slack_cost))

                for s in range(self.network_scenario.num_apps):
                    for j in range(self.network_scenario.num_DCs):
                        if t == (self.num_time_slots - 1):
                            index.extend([mu[s][j][t]])
                            value.extend([-x_sjt[s][j]+on_sjt[s][j]])
                        else:
                            index.extend([mu[s][j][t+1], mu[s][j][t]])
                            value.extend([x_sjt[s][j], -x_sjt[s][j]+on_sjt[s][j]])
                        rhs_value = rhs_value + x_sjt[s][j]
                        rhs_value = rhs_value + on_sjt[s][j]

                theta_constraint = cplex.SparsePair(ind=index, val=value)

                model.linear_constraints.add(lin_expr=[theta_constraint],
                                             senses=["L"],
                                             rhs=[rhs_value])

        # Solve model
        model.objective.set_sense(model.objective.sense.maximize)
        try:
            model.set_log_stream(None)
            model.set_results_stream(None)
            model.parameters.timelimit.set(7200)  # two hours
            start_time = time.time()
            model.solve()
            end_time = time.time()
        except CplexSolverError as e:
            print("Exception raised during solve: {}".format(e))
            return
        solution = model.solution
        print("Master problem. Time taken = {} seconds".format(end_time-start_time))
        solution_values = model.solution.get_values()

        mu_vals = []
        for s in range(self.network_scenario.num_apps):
            mu_vals.append([])
            for j in range(self.network_scenario.num_DCs):
                mu_vals[s].append([])
                for t in range(self.num_time_slots):
                    mu_vals[s][j].append(solution.get_values(mu[s][j][t]))
        obj = solution.get_objective_value()

        return obj, mu_vals

