import random
import numpy as np
import matplotlib.pyplot as plt

class NetworkScenario:
    def __init__(self, time_slot_in_seconds, num_BSs, num_DCs, num_apps, demand, app_groups,
                 start_time_index, latency_cloud, latency_DC, compute_capacities, mem_capacities,
                 resilience_per_app):
        self.num_BSs = num_BSs
        self.num_DCs = num_DCs
        self.num_apps = num_apps
        self.time_slot_in_seconds = time_slot_in_seconds
        self.app_groups = app_groups
        self.start_time_index = start_time_index
        self.latency_cloud = latency_cloud
        self.latency_DC = latency_DC
        self.compute_capacities = compute_capacities
        self.mem_capacities = mem_capacities
        self.resilience_per_app = resilience_per_app

        # Costs in the objective function
        self.fixed_cost = [1000 for j in range(self.num_DCs)]
        self.op_cost = [350 for j in range(self.num_DCs)]
        self.slack_cost = [200 for s in range(self.num_apps)]
        self.cloud_cost = [200 for s in range(self.num_apps)]

        self.demand = demand

        # Keep application requirements unchanged for both random scenarios and use cases
        self.max_latency_per_app = {"Video": 50,  # 70
                                    "Compute": 20,
                                    "AR": 20,
                                    "Health": 40}  # 100
        self.compute_factor_per_app = {"Video": 3,
                                       "Compute": 433,
                                       "AR": 8,
                                       "Health": 1}
        self.model_loading_per_app = {"Video": 15,
                                      "Compute": 10,
                                      "AR": 15,
                                      "Health": 2}
        self.memory_per_app = {"Video": 1192,
                               "Compute": 1100,
                               "AR": 1320,
                               "Health": 1240}
        self.input_size_per_app = {"Video": 0.4,
                                   "Compute": 1,
                                   "AR": 0.6,
                                   "Health": 0.3}

        self.resilience = [1 for s in range(self.num_apps)]
        self.max_latency = [0 for s in range(self.num_apps)]
        self.compute_factor = [0 for s in range(self.num_apps)]
        self.mem_factor = [0 for s in range(self.num_apps)]
        self.input_size = [0 for s in range(self.num_apps)]
        self.model_loading_latency = [0 for s in range(self.num_apps)]
        self.latency_priority = [0 for s in range(self.num_apps)]
        #self.model_load_energy = [random.randint(10,15) for s in range(self.num_apps)]
        self.model_load_energy = [12, 10, 10, 12, 15, 10, 15, 15, 15, 12, 15, 12, 15, 15, 11, 10, 12, 14, 13, 11] #cost_1
        #self.model_first_download_energy = [random.randint(20,30) for s in range(self.num_apps)]
        self.model_first_download_energy = [28, 26, 21, 21, 25, 29, 24, 23, 29, 22, 22, 22, 30, 28, 30, 24, 22, 21, 27, 20] # cost_2

        for s in range(self.num_apps):
            for key, value in self.app_groups.items():
                if s in value:
                    self.max_latency[s] = self.max_latency_per_app[key]
                    self.compute_factor[s] = self.compute_factor_per_app[key]
                    self.mem_factor[s] = self.memory_per_app[key]
                    self.input_size[s] = self.input_size_per_app[key]
                    self.model_loading_latency[s] = self.model_loading_per_app[key]
                    self.resilience[s] = self.resilience_per_app[key]
                    if key == "Compute" or key == "AR":
                        self.latency_priority[s] = 1

class SolutionObject:
    def __init__(self, scenario, objective_value, num_DCs_open,
                 w_isj_solution, z_j_solution, x_sj_solution, on_sj_solution, slack_variable):
        self.scenario = scenario
        self.objective_value = objective_value
        self.num_DCs_open = num_DCs_open
        self.fraction_assigned = w_isj_solution
        self.DCs_open = z_j_solution
        self.apps_loaded = x_sj_solution
        self.apps_loaded_curr_timeslot = on_sj_solution
        self.latency_exceed = slack_variable

    def calculate_total_requests_on_DCs(self, time_slot_index):
        total_requests_DC = [0 for _ in range(self.scenario.num_DCs)]
        for j in range(self.scenario.num_DCs):
            total_requests_DC[j] = 0
            for s in range(self.scenario.num_apps):
                for i in range(self.scenario.num_BSs):
                    total_requests_DC[j] += self.scenario.demand[i][s][time_slot_index] * \
                               self.fraction_assigned[i][s][j+1]
        return total_requests_DC

    def calculate_demand_on_cloud(self, time_slot_index):
        total_demand_cloud = 0
        for i in range(self.scenario.num_BSs):
            for s in range(self.scenario.num_apps):
                total_demand_cloud += self.fraction_assigned[i][s][0] * self.scenario.demand[i][s][time_slot_index]
        return total_demand_cloud

    def calculate_utilization_DCs(self, time_slot_index):
        util = [0 for _ in range(self.scenario.num_DCs)]
        for j in range(self.scenario.num_DCs):
            util[j] = 0
            for s in range(self.scenario.num_apps):
                for i in range(self.scenario.num_BSs):
                    util[j] += self.scenario.compute_factor[s] * self.scenario.demand[i][s][time_slot_index] * \
                               self.fraction_assigned[i][s][
                                   j + 1]
            util[j] = util[j] / self.scenario.compute_capacities[j]
        return util

    def calculate_objective_with_fixed_load_costs(self, time_slot_index):
        utilization_per_DC = self.calculate_utilization_DCs(time_slot_index)
        energy_cost = 0
        latency_cost = 0
        cloud_cost = 0
        apps_loaded = 0
        apps_loaded_first = 0
        for j in range(self.scenario.num_DCs):
            if self.DCs_open[j] > 0:
                energy_cost = energy_cost + self.scenario.fixed_cost[j]
            energy_cost = energy_cost + self.scenario.op_cost[j] * utilization_per_DC[j]

            # Cost for loading apps
            x_sj = np.array(self.apps_loaded)
            energy_cost = energy_cost + np.sum(x_sj[:, j])
            apps_loaded += np.sum(x_sj[:,j])
            # Calculate energy required for loading app in this time slot
            on_sj = np.array(self.apps_loaded_curr_timeslot)
            energy_cost = energy_cost + np.sum(on_sj[:, j])
            apps_loaded_first += np.sum(on_sj[:,j])
        # Cost for assigning to cloud
        for i in range(self.scenario.num_BSs):
            for s in range(self.scenario.num_apps):
                cloud_cost = cloud_cost + self.scenario.cloud_cost[s] * self.fraction_assigned[i][s][0]

        # Cost for exceeding latency
        avg_latencies = self.calculate_average_latency(time_slot_index)
        for s in range(self.scenario.num_apps):
            if avg_latencies[s] > self.scenario.max_latency[s]:
                latency_cost = latency_cost + self.scenario.slack_cost[s] * (avg_latencies[s]-self.scenario.max_latency[s])
        objective = energy_cost + latency_cost + cloud_cost
        print("Time slot = {}, overall apps_loaded = {}, overall apps_loaded_first = {}".format(time_slot_index, apps_loaded, apps_loaded_first))
        return objective, energy_cost, latency_cost, cloud_cost

    def calculate_energy_DCs(self, time_slot_index):
        utilization_per_DC = self.calculate_utilization_DCs(time_slot_index)
        energy = [0 for _ in range(self.scenario.num_DCs)]
        for j in range(self.scenario.num_DCs):
            if np.isclose(self.DCs_open[j], 1):
                energy[j] = energy[j] + 200
                energy[j] = energy[j] + 350 * utilization_per_DC[j]
                # Calculate energy required for loading apps with cost=1
                x_sj = np.array(self.apps_loaded)
                energy[j] = energy[j] + np.sum(x_sj[:,j])

                # Calculate energy required for loading app in this time slot with cost=1
                on_sj = np.array(self.apps_loaded_curr_timeslot)
                energy[j] = energy[j] + np.sum(on_sj[:, j])
        return energy, utilization_per_DC

    def print_app_loaded_per_DC(self, time_slot_index):
        apps_loaded_per_DC = [[] for j in range(self.scenario.num_DCs)]
        apps_loaded_curr_time_per_DC = [[] for j in range(self.scenario.num_DCs)]
        app_load_count = [0 for s in range(self.scenario.num_apps)]

        for j in range(self.scenario.num_DCs):
            for s in range(self.scenario.num_apps):
                if self.apps_loaded[s][j] > 0:
                    app_load_count[s] = app_load_count[s] + 1
                    apps_loaded_per_DC[j].append(s)
                    # app is loaded and in this time slot
                    if self.apps_loaded_curr_timeslot[s][j] > 0:
                        apps_loaded_curr_time_per_DC[j].append(s)
            print("Time slot = {}, DC {}. Apps loaded  = {}, Apps loaded in this time slot = {}. "
                  "Number of apps loaded = {}, and in this time slot = {}".format(time_slot_index,
                                                                                  j,
                                                                                  apps_loaded_per_DC[j],
                                                                                  apps_loaded_curr_time_per_DC[j],
                                                                                  len(apps_loaded_per_DC[j]),
                                                                                  len(apps_loaded_curr_time_per_DC[j]),
                                                                                  ))
        for s in range(self.scenario.num_apps):
            print("Time slot = {}, App {} loaded in {} DCs".format(time_slot_index, s, app_load_count[s]))

    def calculate_memory_utilization(self, time_slot_index):
        memory = [0 for j in range(self.scenario.num_DCs)]
        for j in range(self.scenario.num_DCs):
            memory_workload = 0
            for s in range(self.scenario.num_apps):
                # Memory consumed by loading models
                if self.apps_loaded[s][j] == 1:
                    memory[j] = memory[j] + self.scenario.mem_factor[s]

                for i in range(self.scenario.num_BSs):
                    memory_workload = memory_workload + (self.fraction_assigned[i][s][j + 1] *
                                                         self.scenario.demand[i][s][time_slot_index] *
                                                         self.scenario.input_size[s])
            memory_workload = memory_workload / self.scenario.time_slot_in_seconds
            memory[j] = memory[j] + memory_workload
            memory[j] = memory[j] / self.scenario.mem_capacities[j]
        return memory

    def calculate_average_latency(self, time_slot_index):
        latency = [0 for s in range(self.scenario.num_apps)]
        for s in range(self.scenario.num_apps):
            total_demand_for_s = sum(
                [self.scenario.demand[i][s][time_slot_index] for i in range(self.scenario.num_BSs)])
            if total_demand_for_s > 0:
                for i in range(self.scenario.num_BSs):
                    # latency of requests that go to the cloud
                    latency[s] = latency[s] + (
                            self.fraction_assigned[i][s][0] * self.scenario.demand[i][s][time_slot_index] *
                            self.scenario.latency_cloud[i])

                    for j in range(self.scenario.num_DCs):
                        # model loading latency
                        if self.apps_loaded[s][j] > 0 and self.apps_loaded_curr_timeslot[s][j] > 0:
                            latency[s] = latency[s] + (
                                    self.fraction_assigned[i][s][j + 1] * self.scenario.model_loading_latency[s] *
                                    self.scenario.demand[i][s][time_slot_index] / self.scenario.time_slot_in_seconds)

                        # latency of reaching each DC
                        latency[s] = latency[s] + (
                                self.fraction_assigned[i][s][j + 1] * self.scenario.latency_DC[i][j] *
                                self.scenario.demand[i][s][time_slot_index])
                latency[s] = latency[s] / total_demand_for_s
            #print("Average latency for application {} = {}, demand = {}".format(s, latency[s], total_demand_for_s))
        return latency

    def analyze_fractions_assigned(self, time_slot_index):
        demand_assigned = np.array(
            [[0.0 for j in range(self.scenario.num_DCs + 1)] for i in range(self.scenario.num_BSs)])

        for s in range(self.scenario.num_apps):
            total_demand = sum([self.scenario.demand[i][s][time_slot_index] for i in range(self.scenario.num_BSs)])
            demand_to_cloud = 0
            for i in range(self.scenario.num_BSs):
                demand_to_cloud += (self.fraction_assigned[i][s][0] * self.scenario.demand[i][s][time_slot_index])
                print("Demand assignment for BS {} = {}".format(i, self.fraction_assigned[i][s]))
                for j in range(self.scenario.num_DCs + 1):
                    demand_assigned[i][j] = self.fraction_assigned[i][s][
                        j]  # * self.scenario.demand[i][s][time_slot_index]
            print("Application {}, demand assigned to cloud = {}".format(s, demand_to_cloud))


    def calculate_cost_on_sj(self, cost, num_DCs):
        on_sj = np.array(self.apps_loaded_curr_timeslot)
        total_cost = 0
        for j in range(num_DCs):
            cost_per_DC = np.sum(on_sj[:,j] * cost)
            total_cost = total_cost + cost_per_DC
        print("Number of apps loaded in this time slot = {}".format(np.sum(on_sj)))
        return total_cost

    def calculate_cost_x_sj(self, cost, num_DCs):
        x_sj = np.array(self.apps_loaded)
        total_cost = 0
        for j in range(num_DCs):
            cost_per_DC = np.sum(x_sj[:,j] * cost)
            total_cost = total_cost + cost_per_DC
        print("Number of apps loaded = {}".format(np.sum(x_sj)))
        return total_cost

    def check_slack_values(self):
        for s in range(self.scenario.num_apps):
            if self.latency_exceed[s] > 0:
                print("Latency exceeded for app {} by {}".format(s, self.latency_exceed[s]))