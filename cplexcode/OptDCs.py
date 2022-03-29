import cplex
from cplex.exceptions import CplexSolverError
import sys
from class_definitions import NetworkScenario
import time

class OptDCs:
    def __init__(self):
        pass

    # Number of time slots are determined by shape of demand vector
    def opt_dcs(self, scenario, start_time_index=0, with_benders=True,
                with_lagrangian=False, mu=None, last_index=False, lagrangian_iter=None):
        num_time_slots = len(scenario.demand[0][0])

        total_compute_entities = scenario.num_DCs + 1
        bigM_1 = scenario.num_BSs
        bigM_2 = scenario.num_apps # TODO: check with num_BSs

        # Create a new (empty) model and populate it below.
        model = cplex.Cplex()

        # Create one continuous variable w for each base station i, application s, DC j
        # index j=0 refers to the cloud
        for i in range(scenario.num_BSs):
            for s in range(scenario.num_apps):
                for j in range(total_compute_entities):
                    for t in range(num_time_slots):
                        if j == 0:
                            objective_factor = [scenario.cloud_cost[s]]
                        else:
                            objective_factor = [
                                scenario.op_cost[j - 1] * scenario.compute_factor[s] * scenario.demand[i][s][t] / scenario.compute_capacities[j - 1]]
                        model.variables.add(obj=objective_factor,
                                            lb=[0],
                                            ub=[1.0],
                                            types=["C"],
                                            names=["w_{}_{}_{}_{}".format(i, s, j, t)])  # w_i,s,j,t

        # Create one binary variable for each DC/application pair. The variables
        # model whether a DC j has application s running on it: x[s][j]
        # objective function has higher cost for higher power level
        for s in range(scenario.num_apps):
            for j in range(scenario.num_DCs):
                if with_lagrangian:
                    if last_index:
                        model.variables.add(obj=[1 + mu[s][j][t] for t in range(num_time_slots)],
                                            lb=[0] * num_time_slots,
                                            ub=[1] * num_time_slots,
                                            types=["B"] * num_time_slots,
                                            names=["x_{}_{}_{}".format(s, j, t) for t in range(num_time_slots)])  # x_s,j,t
                    else:
                        # print(s, j, [scenario.model_load_energy[s] - mu[s][j][t+1] + mu[s][j][t] for t in range(num_time_slots)])
                        model.variables.add(obj=[1 - mu[s][j][t+1] + mu[s][j][t] for t in range(num_time_slots)],
                                            lb=[0] * num_time_slots,
                                            ub=[1] * num_time_slots,
                                            types=["B"] * num_time_slots,
                                            names=["x_{}_{}_{}".format(s, j, t) for t in range(num_time_slots)])  # x_s,j,t

                else:
                    model.variables.add(obj=[15] * num_time_slots,
                                        lb=[0] * num_time_slots,
                                        ub=[1] * num_time_slots,
                                        types=["B"] * num_time_slots,
                                        names=["x_{}_{}_{}".format(s, j, t) for t in range(num_time_slots)])  # x_s,j,t



        # Create one binary variable for each cell. The variables model
        # whether each DC z is switched on or not
        # type B = model.variables.type.binary
        for j in range(scenario.num_DCs):
            for t in range(num_time_slots):
                model.variables.add(obj=[scenario.fixed_cost[j]],
                                    lb=[0],
                                    ub=[1],
                                    types=["B"],
                                    names=["z_{}_{}".format(j, t)])  # z_j,t

        # Create one binary variable for each DC/application pair. The variables
        # model whether a DC j has to load application s in the current time slot
        for s in range(scenario.num_apps):
            for j in range(scenario.num_DCs):
                if with_lagrangian:
                    model.variables.add(obj=[1-mu[s][j][t] for t in range(num_time_slots)],
                                        lb=[0] * num_time_slots,
                                        ub=[1] * num_time_slots,
                                        types=["B"] * num_time_slots,
                                        names=["on_{}_{}_{}".format(s, j, t) for t in range(num_time_slots)])  # ON_s,j,t
                else:
                    model.variables.add(obj=[30] * num_time_slots,
                                        lb=[0] * num_time_slots,
                                        ub=[1] * num_time_slots,
                                        types=["B"] * num_time_slots,
                                        names=["on_{}_{}_{}".format(s, j, t) for t in range(num_time_slots)])  # ON_s,j,t


        # Create one continuous variable for each BS/DC/application. The variable
        # model whether the demand from BS i for application s is handled by
        # an application that is loaded in DC j in this time slot (based on ON_s,j)
        for i in range(scenario.num_BSs):
            for s in range(scenario.num_apps):
                for j in range(scenario.num_DCs):
                    model.variables.add(obj=[0] * num_time_slots,
                                        lb=[0] * num_time_slots,
                                        ub=[1] * num_time_slots,
                                        types=["C"] * num_time_slots,
                                        names=["delta_{}_{}_{}_{}".format(i, s, j, t) for t in
                                               range(num_time_slots)])  # delta_isjt

        # Create one continuous variable for each application. The variable
        # model whether the latency of serving requests for application s
        # can be exceeded by a certain amount
        for s in range(scenario.num_apps):
            model.variables.add(obj=[scenario.slack_cost[s]] * num_time_slots,
                                lb=[0] * num_time_slots,
                                ub=[100] * num_time_slots,
                                types=["C"] * num_time_slots,
                                names=["gamma_{}_{}".format(s, t) for t in range(num_time_slots)])  # gamma_s,t

        index = 0
        # Create w indices for later use
        w = []
        for i in range(scenario.num_BSs):
            w.append([])
            for s in range(scenario.num_apps):
                w[i].append([])
                for j in range(total_compute_entities):
                    w[i][s].append([])
                    for t in range(num_time_slots):
                        w[i][s][j].append(index)
                        index += 1
        # print("W indices = {}".format(w))

        # Create x indices for later use
        x = []
        for s in range(scenario.num_apps):
            x.append([])
            for j in range(scenario.num_DCs):
                x[s].append([])
                for t in range(num_time_slots):
                    x[s][j].append(index)
                    index += 1
        # print("x indices = {}".format(x))

        # Create z indices for later use
        z = []
        for j in range(scenario.num_DCs):
            z.append([])
            for t in range(num_time_slots):
                z[j].append(index)
                index += 1
        # print("z indices = {}".format(z))

        # Create ON indices for later use
        on = []
        for s in range(scenario.num_apps):
            on.append([])
            for j in range(scenario.num_DCs):
                on[s].append([])
                for t in range(num_time_slots):
                    on[s][j].append(index)
                    index += 1
        # print("on indices = {}".format(on))

        # Create delta indices for later use
        delta = []
        for i in range(scenario.num_BSs):
            delta.append([])
            for s in range(scenario.num_apps):
                delta[i].append([])
                for j in range(scenario.num_DCs):
                    delta[i][s].append([])
                    for t in range(num_time_slots):
                        delta[i][s][j].append(index)
                        index += 1
        # print("Delta indices = {}".format(delta))

        # Create gamma indices for later use
        gamma = []
        for s in range(scenario.num_apps):
            gamma.append([])
            for t in range(num_time_slots):
                gamma[s].append(index)
                index += 1
        # print("Gamma indices = {}".format(gamma))

        # Set ON variable if model s is loaded in DC j in current time slot
        if not with_lagrangian:
            for s in range(scenario.num_apps):
                for j in range(scenario.num_DCs):
                    index = [x[s][j][0], on[s][j][0]]
                    value = [1, -1]
                    assign_on_constraint_initial = cplex.SparsePair(ind=index, val=value)
                    model.linear_constraints.add(lin_expr=[assign_on_constraint_initial],
                                                 senses=["E"],
                                                 rhs=[0.0])

                    for t in range(1, num_time_slots):
                        index = [x[s][j][t- 1], x[s][j][t], on[s][j][t]]
                        value = [-1, 1, -1]
                        assign_on_constraint = cplex.SparsePair(ind=index, val=value)
                        model.linear_constraints.add(lin_expr=[assign_on_constraint],
                                                     senses=["L"],
                                                     rhs=[0.0])

        # Latency constraint
        latency_names = ["latency{}".format(latency_index) for latency_index in
                         range(num_time_slots * scenario.num_apps)]
        constraint_index = 0
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                total_demand = sum([scenario.demand[i][s][t] for i in range(scenario.num_BSs)])  # in this time slot
                index = []
                value = []
                if total_demand > 0:
                    for i in range(scenario.num_BSs):
                        index.append(w[i][s][0][t])
                        value.append(scenario.latency_cloud[i] * scenario.demand[i][s][t] / total_demand)

                        index.extend([w[i][s][j + 1][t] for j in range(scenario.num_DCs)
                                      ])
                        value.extend([scenario.latency_DC[i][j] * scenario.demand[i][s][t] / total_demand for j in range(scenario.num_DCs)])

                        index.extend([delta[i][s][j][t] for j in range(scenario.num_DCs)])
                        value.extend(
                            [scenario.model_loading_latency[s] * scenario.demand[i][s][t] / total_demand / scenario.time_slot_in_seconds for j in
                             range(scenario.num_DCs)])

                    index.append(gamma[s][t])
                    value.append(-1.0)

                latency_constraint = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[latency_constraint],
                                             senses=["L"],
                                             rhs=[scenario.max_latency[s]],
                                             names=[latency_names[constraint_index]])
                constraint_index = constraint_index + 1

        # Resilience constraint
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                index = [x[s][j][t] for j in range(scenario.num_DCs)]
                value = [1.0] * scenario.num_DCs
                resilience_constraint = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[resilience_constraint],
                                             senses=["G"],
                                             rhs=[scenario.resilience[s]])

        # Capacity constraint for computation
        for t in range(num_time_slots):
            for j in range(1, total_compute_entities):  # only for DCs
                index = []
                value = []
                for s in range(scenario.num_apps):
                    index.extend([w[i][s][j][t] for i in range(scenario.num_BSs)])
                    value.extend([scenario.demand[i][s][t] * scenario.compute_factor[s] for i in range(scenario.num_BSs)])
                capacity_constraint = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[capacity_constraint],
                                             senses=["L"],
                                             rhs=[0.7 * scenario.compute_capacities[j - 1]])

        # Capacity constraint for loading models
        mem_load_names = ["mem_load{}".format(mem_load_index) for mem_load_index in
                          range(num_time_slots * scenario.num_DCs)]
        constraint_index = 0
        for t in range(num_time_slots):
            for j in range(scenario.num_DCs):
                index = [x[s][j][t] for s in range(scenario.num_apps)]
                value = [scenario.mem_factor[s] for s in range(scenario.num_apps)]

                mem_load_constraint = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[mem_load_constraint],
                                             senses=["L"],
                                             rhs=[0.7 * scenario.mem_capacities[j]],
                                             names=[mem_load_names[constraint_index]])
                constraint_index = constraint_index + 1

        # Capacity constraint for memory
        mem_names = ["mem{}".format(mem_load_index) for mem_load_index in
                     range(num_time_slots * scenario.num_DCs)]
        constraint_index = 0
        for t in range(num_time_slots):
            for j in range(scenario.num_DCs):
                index = [x[s][j][t] for s in range(scenario.num_apps)]
                value = [scenario.mem_factor[s] for s in range(scenario.num_apps)]

                for s in range(scenario.num_apps):
                    index.extend([w[i][s][j + 1][t] for i in range(scenario.num_BSs)])
                    value.extend([scenario.demand[i][s][t] * scenario.input_size[s] / scenario.time_slot_in_seconds for i in range(scenario.num_BSs)])

                mem_capacity_constraint = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[mem_capacity_constraint],
                                             senses=["L"],
                                             rhs=[0.95 * scenario.mem_capacities[j]],
                                             names=[mem_names[constraint_index]])
                constraint_index = constraint_index + 1

        # Demand is assigned to at least one location
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                for i in range(scenario.num_BSs):
                    index = [w[i][s][j][t] for j in range(total_compute_entities)]
                    value = [scenario.demand[i][s][t]] * total_compute_entities
                    demand_constraint = cplex.SparsePair(ind=index, val=value)
                    model.linear_constraints.add(lin_expr=[demand_constraint],
                                                 senses=["E"],
                                                 rhs=[1.0 * scenario.demand[i][s][t]])

        # Whether application s is running on DC j
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                for j in range(scenario.num_DCs):
                    index = [w[i][s][j + 1][t] for i in range(scenario.num_BSs)]
                    value = [1.0] * scenario.num_BSs
                    index.append(x[s][j][t])
                    value.append(-bigM_1)

                    assign_binary_constraint_x = cplex.SparsePair(ind=index, val=value)
                    model.linear_constraints.add(lin_expr=[assign_binary_constraint_x],
                                                 senses=["L"],
                                                 rhs=[0.0])

        # Whether DC j is on or off
        for t in range(num_time_slots):
            for j in range(scenario.num_DCs):
                index = [x[s][j][t] for s in range(scenario.num_apps)]
                value = [1.0] * scenario.num_apps

                index.append(z[j][t])
                value.append(-bigM_2)

                assign_binary_constraint_z = cplex.SparsePair(ind=index, val=value)
                model.linear_constraints.add(lin_expr=[assign_binary_constraint_z],
                                             senses=["L"],
                                             rhs=[0.0])

        # Set delta[i][s][j] to w[i][s][j] when ON[s][j]=1
        for t in range(num_time_slots):
            for i in range(scenario.num_BSs):
                for s in range(scenario.num_apps):
                    for j in range(scenario.num_DCs):
                        index = [delta[i][s][j][t], w[i][s][j + 1][t]]
                        value = [1.0, -1.0]
                        delta_1_constraint = cplex.SparsePair(ind=index, val=value)
                        model.linear_constraints.add(lin_expr=[delta_1_constraint],
                                                     senses=["L"],
                                                     rhs=[0.0])

                        index = [delta[i][s][j][t], w[i][s][j + 1][t], on[s][j][t]]
                        value = [-1.0, 1.0, 1.0]
                        delta_2_constraint = cplex.SparsePair(ind=index, val=value)
                        model.linear_constraints.add(lin_expr=[delta_2_constraint],
                                                     senses=["L"],
                                                     rhs=[1.0])

                        index = [delta[i][s][j][t], on[s][j][t]]
                        value = [1.0, -1.0]
                        delta_3_constraint = cplex.SparsePair(ind=index, val=value)
                        model.linear_constraints.add(lin_expr=[delta_3_constraint],
                                                     senses=["L"],
                                                     rhs=[0.0])


        # Our objective is to minimize cost. Costs have been set when variables were created.
        model.objective.set_sense(model.objective.sense.minimize)

        # # Solve
        try:
            if with_lagrangian:
                if lagrangian_iter <= 5:
                    model.parameters.mip.tolerances.mipgap.set(0.01)
                model.parameters.timelimit.set(1200)  # twenty minutes
            else:
                model.parameters.timelimit.set(18000)  # five hours

            if with_benders:
                model.parameters.benders.strategy.set(3)
            model.set_log_stream(None)
            model.set_results_stream(None)
            time_start = time.time()
            model.solve()
            end_time = time.time()

        except CplexSolverError as e:
            print("Exception raised during solve: {}".format(e))
            return

        solution = model.solution

        if with_lagrangian:
            print("Time slot = {}. Solution status with code {} = {}. Time taken = {} seconds".format(start_time_index,
                                                                             solution.get_status(),
                                                                             solution.status[solution.get_status()],
                                                                             end_time - time_start))
        else:
            print("Solution status with code {} = {}".format(solution.get_status(),
                                                             solution.status[solution.get_status()]))
        if solution.get_status() == 103:
            model.conflict.refine(model.conflict.all_constraints())
            model.conflict.write("conflict.clp")
            return

        w_tisj_solution = [
            [[[0 for j in range(total_compute_entities)] for s in range(scenario.num_apps)] for i in range(scenario.num_BSs)] for
            t in range(num_time_slots)]
        for t in range(num_time_slots):
            for i in range(scenario.num_BSs):
                for s in range(scenario.num_apps):
                    for j in range(total_compute_entities):
                        if solution.get_values(w[i][s][j][t]) > model.parameters.mip.tolerances.integrality.get():
                            w_tisj_solution[t][i][s][j] = solution.get_values(w[i][s][j][t])

    
        on_tsj_solution = [[[0 for j in range(scenario.num_DCs)] for s in range(scenario.num_apps)] for t in
                           range(num_time_slots)]
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                for j in range(scenario.num_DCs):
                    if solution.get_values(on[s][j][t]) > model.parameters.mip.tolerances.integrality.get():
                        on_tsj_solution[t][s][j] = solution.get_values(on[s][j][t])

        scenario.num_DCs_open = [0 for t in range(num_time_slots)]
        z_tj_solution = [[0 for j in range(scenario.num_DCs)] for t in range(num_time_slots)]
        for t in range(num_time_slots):
            for j in range(scenario.num_DCs):
                z_tj_solution[t][j] = solution.get_values(z[j][t])
                if z_tj_solution[t][j] > model.parameters.mip.tolerances.integrality.get():
                    scenario.num_DCs_open[t] += 1

        x_tsj_solution = [[[0 for j in range(scenario.num_DCs)] for s in range(scenario.num_apps)] for t in range(num_time_slots)]
        for t in range(num_time_slots):
            for s in range(scenario.num_apps):
                for j in range(scenario.num_DCs):
                    if solution.get_values(x[s][j][t]) > model.parameters.mip.tolerances.integrality.get():
                        x_tsj_solution[t][s][j] = solution.get_values(x[s][j][t])

        slack_ts_solution = [[0 for s in range(scenario.num_apps)] for t in range(num_time_slots)]
        for s in range(scenario.num_apps):
            for t in range(num_time_slots):
                if solution.get_values(gamma[s][t]) > model.parameters.mip.tolerances.integrality.get():
                    slack_ts_solution[t][s] = solution.get_values(gamma[s][t])

        return solution.get_objective_value(), scenario.num_DCs_open, w_tisj_solution, z_tj_solution, x_tsj_solution, \
               on_tsj_solution, slack_ts_solution
