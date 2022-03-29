import math
from collections import Counter
import numpy as np


def greedy_solution_bfd_baseline_capacity(scenario, time_slot_index, prev_open_DCs, prev_loaded_apps):
    apps_loaded = [[0 for j in range(scenario.num_DCs)] for s in range(scenario.num_apps)]
    orig_available_compute_DCs = np.copy(scenario.compute_capacities) * 0.7  # limiting to 70% utilization
    orig_available_memory_DCs = np.copy(scenario.mem_capacities) * 0.95  # limiting memory to 95

    current_available_compute_DCs = np.copy(orig_available_compute_DCs)
    current_available_memory_DCs = np.copy(orig_available_memory_DCs)

    # Index 0 for cloud
    fraction_assigned = [[[0 for j in range(scenario.num_DCs + 1)] for s in range(scenario.num_apps)] for i in
                         range(scenario.num_BSs)]
    demand = np.array(scenario.demand)
    z_j = [0 for _ in range(scenario.num_DCs)]

    # Loop through all applications starting with the app with lowest latency requirement
    app_latencies = [scenario.max_latency[s] for s in range(scenario.num_apps)]
    sorted_app_indices = np.argsort(app_latencies)
    for app_index in sorted_app_indices:
        demand_per_BS = demand[:, app_index, time_slot_index]
        # Loop through each BS, starting with highest demand
        base_station_indices = np.argsort(demand_per_BS)[::-1]
        for i in base_station_indices:
            orig_demand_to_satisfy = demand[i][app_index][time_slot_index]
            demand_to_satisfy = demand[i][app_index][time_slot_index]
            if demand_to_satisfy == 0:
                continue
            memory_reqd = [0 for _ in range(scenario.num_DCs)]
            compute_reqd = [0 for _ in range(scenario.num_DCs)]
            demand_to_assign = [0 for _ in range(scenario.num_DCs)]
            remaining_capacity = np.copy(current_available_compute_DCs)
            # Calculate remaining memory capacity after packing demand
            # for app s from BS i to DC_index
            for DC_index in range(scenario.num_DCs):
                min_compute = min(current_available_compute_DCs[DC_index],
                                  round(scenario.compute_factor[app_index] * demand_to_satisfy, 2))
                demand_that_fits = min_compute // scenario.compute_factor[app_index]
                compute_reqd[DC_index] = min_compute
                if apps_loaded[app_index][DC_index] == 1:
                    # App is already loaded, only require memory for processing data
                    min_memory = min(current_available_memory_DCs[DC_index],
                                     demand_to_satisfy * scenario.input_size[app_index] / scenario.time_slot_in_seconds)
                    demand_to_assign[DC_index] = min(demand_that_fits,
                                                     round(min_memory * scenario.time_slot_in_seconds / scenario.input_size[
                                                            app_index], 0))
                    memory_reqd[DC_index] = demand_to_assign[DC_index] * \
                                            scenario.input_size[app_index] / scenario.time_slot_in_seconds
                else:
                    min_memory = min(current_available_memory_DCs[DC_index],
                                     scenario.mem_factor[app_index] + demand_to_satisfy * scenario.input_size[
                                         app_index] / scenario.time_slot_in_seconds)
                    demand_to_assign[DC_index] = min(demand_that_fits, round((min_memory - scenario.mem_factor[
                        app_index]) * scenario.time_slot_in_seconds / scenario.input_size[app_index], 0))
                    memory_reqd[DC_index] = scenario.mem_factor[app_index] + demand_to_assign[DC_index] * scenario.input_size[
                        app_index] / scenario.time_slot_in_seconds
                remaining_capacity[DC_index] = current_available_compute_DCs[DC_index] - (demand_to_assign[DC_index] *
                                                                                         scenario.compute_factor[app_index])
                if remaining_capacity[DC_index] <= 0 or demand_to_assign[DC_index] == 0:
                    remaining_capacity[DC_index] = math.inf

            chosen_DC = np.argmin(remaining_capacity)
            # Assign demand to chosen DC

            fraction = demand_to_assign[chosen_DC] / orig_demand_to_satisfy

            demand_to_satisfy = demand_to_satisfy - demand_to_assign[chosen_DC]
            current_available_memory_DCs[chosen_DC] = current_available_memory_DCs[chosen_DC] - memory_reqd[
                chosen_DC]
            current_available_compute_DCs[chosen_DC] = current_available_compute_DCs[chosen_DC] - compute_reqd[chosen_DC]
            apps_loaded[app_index][chosen_DC] = 1
            fraction_assigned[i][app_index][chosen_DC + 1] = fraction
            z_j[chosen_DC] = 1
            # After checking all DCs, if demand remains assign to the cloud
            if demand_to_satisfy > 0:
                fraction_assigned[i][app_index][0] = demand_to_satisfy / orig_demand_to_satisfy

    compute_utilization_DC = (orig_available_compute_DCs - current_available_compute_DCs) / (
            orig_available_compute_DCs / 0.7)

    for i in range(scenario.num_BSs):
        for s in range(scenario.num_apps):
            if not math.isclose(fraction_assigned[i][s][0], 0, abs_tol=1e-5):
                print("Time slot = {}, i = {}, app = {}, fraction assigned to cloud = {}".format(time_slot_index, i, s,
                                                                                                 fraction_assigned[i][
                                                                                                     s][0]))

    # Check if all demand is assigned
    for i in range(scenario.num_BSs):
        for s in range(scenario.num_apps):
            if demand[i][s][time_slot_index] > 0 and not math.isclose(1.0,
                                                                             np.sum(fraction_assigned[i][s])):
                print("Time slot = {}, All demand for app {} from BS {} not assigned. Fractions = {}".format(time_slot_index,
                                                                                                             s, i,
                                                                                             fraction_assigned[i][s]))
                print("Exception!")
                exit(1)
    DCs_open = (compute_utilization_DC > 0).astype(int)

    return compute_utilization_DC.tolist(), fraction_assigned, np.array(apps_loaded), DCs_open, np.sum(DCs_open)


def greedy_solution_bfd_baseline_latency(scenario, time_slot_index, prev_open_DCs, prev_loaded_apps):
    apps_loaded = [[0 for j in range(scenario.num_DCs)] for s in range(scenario.num_apps)]
    orig_available_compute_DCs = np.copy(scenario.compute_capacities) * 0.7  # limiting to 70% utilization
    orig_available_memory_DCs = np.copy(scenario.mem_capacities) * 0.95  # limiting memory to 95% 30400

    current_available_compute_DCs = np.copy(orig_available_compute_DCs)
    current_available_memory_DCs = np.copy(orig_available_memory_DCs)

    # Index 0 for cloud
    fraction_assigned = [[[0 for j in range(scenario.num_DCs + 1)] for s in range(scenario.num_apps)] for i in
                         range(scenario.num_BSs)]
    demand = np.array(scenario.demand)
    z_j = [0 for _ in range(scenario.num_DCs)]

    app_latencies = [scenario.max_latency[s] for s in range(scenario.num_apps)]
    sorted_app_indices = np.argsort(app_latencies)

    for app_index in sorted_app_indices:
        demand_per_BS = demand[:, app_index, time_slot_index]
        # Loop through each BS, starting with highest demand
        base_station_indices = np.argsort(demand_per_BS)[::-1]
        for i in base_station_indices:
            orig_demand_to_satisfy = demand[i][app_index][time_slot_index]
            demand_to_satisfy = demand[i][app_index][time_slot_index]
            if demand_to_satisfy == 0:
                break
            memory_reqd = [0 for _ in range(scenario.num_DCs)]
            compute_reqd = [0 for _ in range(scenario.num_DCs)]
            demand_to_assign = [0 for _ in range(scenario.num_DCs)]
            remaining_capacity = np.copy(current_available_compute_DCs)
            # Calculate remaining memory capacity after packing demand
            # for app s from BS i to DC_index
            for DC_index in range(scenario.num_DCs):
                min_compute = min(current_available_compute_DCs[DC_index],
                                  round(scenario.compute_factor[app_index] * demand_to_satisfy, 2))
                demand_that_fits = min_compute // scenario.compute_factor[app_index]
                compute_reqd[DC_index] = min_compute
                if apps_loaded[app_index][DC_index] == 1:
                    # App is already loaded, only require memory for processing data
                    min_memory = min(current_available_memory_DCs[DC_index],
                                     demand_to_satisfy * scenario.input_size[app_index] / scenario.time_slot_in_seconds)
                    demand_to_assign[DC_index] = min(demand_that_fits,
                                                     round(min_memory * scenario.time_slot_in_seconds / scenario.input_size[
                                                            app_index], 0))
                    memory_reqd[DC_index] = demand_to_assign[DC_index] * \
                                            scenario.input_size[app_index] / scenario.time_slot_in_seconds
                else:
                    min_memory = min(current_available_memory_DCs[DC_index],
                                     scenario.mem_factor[app_index] + demand_to_satisfy * scenario.input_size[
                                         app_index] / scenario.time_slot_in_seconds)
                    demand_to_assign[DC_index] = min(demand_that_fits, round((min_memory - scenario.mem_factor[
                        app_index]) * scenario.time_slot_in_seconds / scenario.input_size[app_index], 0))
                    memory_reqd[DC_index] = scenario.mem_factor[app_index] + demand_to_assign[DC_index] * scenario.input_size[
                        app_index] / scenario.time_slot_in_seconds
                remaining_capacity[DC_index] = current_available_compute_DCs[DC_index] - (demand_to_assign[DC_index] *
                                                                                         scenario.compute_factor[app_index])
                if remaining_capacity[DC_index] <= 0 or demand_to_assign[DC_index] == 0:
                    remaining_capacity[DC_index] = math.inf

            chosen_DC = np.argmin([scenario.latency_DC[i][idx] for idx in range(scenario.num_DCs)])
            # Assign demand to chosen DC
            fraction = demand_to_assign[chosen_DC] / orig_demand_to_satisfy
            demand_to_satisfy = demand_to_satisfy - demand_to_assign[chosen_DC]
            current_available_memory_DCs[chosen_DC] = current_available_memory_DCs[chosen_DC] - memory_reqd[
                chosen_DC]
            current_available_compute_DCs[chosen_DC] = current_available_compute_DCs[chosen_DC] - compute_reqd[chosen_DC]
            apps_loaded[app_index][chosen_DC] = 1
            fraction_assigned[i][app_index][chosen_DC + 1] = fraction
            z_j[chosen_DC] = 1
            # After checking all DCs, if demand remains assign to the cloud
            if demand_to_satisfy > 0:
                fraction_assigned[i][app_index][0] = demand_to_satisfy / orig_demand_to_satisfy

    compute_utilization_DC = (orig_available_compute_DCs - current_available_compute_DCs) / (
            orig_available_compute_DCs / 0.7)

    for i in range(scenario.num_BSs):
        for s in range(scenario.num_apps):
            if not math.isclose(fraction_assigned[i][s][0], 0, abs_tol=1e-5):
                print("Time slot = {}, i = {}, app = {}, fraction assigned to cloud = {}".format(time_slot_index, i, s,
                                                                                                 fraction_assigned[i][
                                                                                                     s][0]))

    # Check if all demand is assigned
    for i in range(scenario.num_BSs):
        for s in range(scenario.num_apps):
            if demand[i][s][time_slot_index] > 0 and not math.isclose(1.0,
                                                                             np.sum(fraction_assigned[i][s])):
                print("Time slot = {}, All demand for app {} from BS {} not assigned. Fractions = {}".format(time_slot_index,
                                                                                                             s, i,
                                                                                             fraction_assigned[i][s]))
                print("Exception!")
                exit(1)
    DCs_open = (compute_utilization_DC > 0).astype(int)

    return compute_utilization_DC.tolist(), fraction_assigned, np.array(apps_loaded), DCs_open, np.sum(DCs_open)

