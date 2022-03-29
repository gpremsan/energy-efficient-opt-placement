from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from cplexcode.OptDCs import OptDCs
from cplexcode.OptDCsTwoStep import OptDCsTwoStep
from cplexcode.LagrangianMaster import LagrangianMaster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import argparse
import concurrent.futures
from class_definitions import NetworkScenario, SolutionObject
import time
import math
import random
from collections import Counter
import os, ctypes
from heuristics import greedy_solution_bfd_baseline_capacity, greedy_solution_bfd_baseline_latency


def solve_one_step_ahead(scenario, time_slot_index, prev_x_sj_value):
    cplex_prob = OptDCsTwoStep()
    objective_value, num_DCs_open, w_isj_solution, \
    z_j_solution, x_sj_solution, on_sj_solution, \
    slack_variable = cplex_prob.opt_dcs(scenario, time_slot_index, prev_x_sj_value)

    return objective_value, num_DCs_open, w_isj_solution, \
           z_j_solution, x_sj_solution, on_sj_solution, slack_variable


def run_cplex(scenario, time_slot_index=0, with_lagrangian=False, mu=[], last_index=False,
                     lagrangian_iter=None, with_benders=True):
    cplex_prob = OptDCs()
    objective_value, num_DCs_open, w_isj_solution, \
    z_j_solution, x_sj_solution, on_sj_solution, \
    slack_variable = cplex_prob.opt_dcs(scenario, time_slot_index, with_benders,
                                        with_lagrangian, mu, last_index, lagrangian_iter)

    return scenario.start_time_index, objective_value, num_DCs_open, w_isj_solution, \
           z_j_solution, x_sj_solution, on_sj_solution, slack_variable


def run_model_parallel(network_scenarios, num_time_slots, complete_mu, time_slots, iter_counter,
                       with_benders, num_parallel_processes):
    end_index = num_time_slots - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_processes) as executor:
        cplex_future = [executor.submit(run_cplex,
                                        network_scenarios[t],
                                        time_slot_index=time_slots[t],
                                        with_lagrangian=True,
                                        mu=complete_mu[:, :, t:t + 2].tolist(),
                                        last_index=False,
                                        lagrangian_iter=iter_counter,
                                        with_benders=with_benders) for t in range(num_time_slots - 1)]

        cplex_future.append(executor.submit(run_cplex,
                                            network_scenarios[end_index],
                                            time_slot_index=time_slots[end_index],
                                            with_lagrangian=True,
                                            mu=complete_mu[:, :, end_index:end_index + 1].tolist(),
                                            last_index=True,
                                            lagrangian_iter=iter_counter,
                                            with_benders=with_benders))
        allres = {}
        for future in concurrent.futures.as_completed(cplex_future):
            try:
                result = future.result()
                allres[result[0]] = result[1:]
            except Exception as ex:
                print("Exception when accessing result: {}".format(ex))
    return allres


def compare_fractions_assigned(scenario, fraction1, fraction2):
    for s in range(scenario.num_apps):
        total_demand = sum([scenario.demand[i][s] for i in range(scenario.num_BSs)])
        print("Total demand for application = {}".format(total_demand))
        demand_to_cloud_1 = 0
        demand_to_cloud_2 = 0

        for i in range(scenario.num_BSs):
            demand_to_cloud_1 += fraction1[i][s][0] * scenario.demand[i][s]
            demand_to_cloud_2 += fraction2[i][s][0] * scenario.demand[i][s]
            if fraction1[i][s] != fraction2[i][s]:
                print("BS {}, application {}".format(i, s))
                print(fraction1[i][s])
                print(fraction2[i][s])
        print("Fraction of demand assigned to cloud without preload = {}".format(demand_to_cloud_1 / total_demand))
        print("Fraction of demand assigned to cloud with preload = {}".format(demand_to_cloud_2 / total_demand))


def print_fraction_results(scenario, fractions):
    for i in range(scenario.num_BSs):
        for j in range(scenario.num_DCs + 1):
            for s in range(scenario.num_apps):
                if fractions[i][s][j] > 0.0:
                    if j == 0:
                        ele_string = "Cloud"
                    else:
                        ele_string = "Edge DC"
                    print("{} {} handles fraction={} of demand={} for application {} from BS {}".
                          format(ele_string, j, fractions[i][s][j], scenario.demand[i][s], s, i))


def read_azure_trace_by_hash_function(day, hash_functions, azure_dataset_path):
    day = "{:02d}".format(day)

    # Header for the DF
    lines = "HashFunction,Trigger,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1359,1360,1361,1362,1363,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403,1404,1405,1406,1407,1408,1409,1410,1411,1412,1413,1414,1415,1416,1417,1418,1419,1420,1421,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,1432,1433,1434,1435,1436,1437,1438,1439,1440\n"
    for hash_function in hash_functions:
        # Change day from which trace is obtained
        filename = azure_dataset_path + "invocations_per_function_md.anon.d{}.csv".format(day)

        with open(filename, "r") as f:
            data = f.read()

        match = re.search("{}.*\n".format(hash_function), data)
        if match:
            lines = lines + match.group(0) + "\n"

    df = pd.read_csv(io.StringIO(lines))

    return df

def calculate_demand_vector_varying_busyness(df, num_time_slots, num_apps, num_BSs, time_slot_in_minutes):
    demand_vector = np.array([[[0 for t in range(num_time_slots)]
                               for s in range(num_apps)] for i in range(num_BSs)])
    time_column_names = list(map(str, range(1, 1441)))
    column_time_indices = [df.columns.get_loc(column_name) for column_name in time_column_names]
    idx = pd.date_range("2021-01-01", periods=1440, freq="T")

    pandas_string = "{}T".format(time_slot_in_minutes)
    # for each application
    for row_idx in range(num_apps):
        ts = pd.Series(df.iloc[row_idx, column_time_indices].values.tolist(), index=idx)
        ts_resampled = ts.resample(pandas_string).sum()

        for t in range(len(ts_resampled)):
            # Decide number of busy, moderately busy and less busy BSs
            num_busy_BSs = random.randint(3, 8)
            num_moderate_BSs = random.randint(1, 8)
            num_less_busy_BSs = num_BSs - num_moderate_BSs - num_busy_BSs

            list_BSs = [*range(0, num_BSs)]
            busy_BSs = random.sample(list_BSs, num_busy_BSs)
            list_BSs = list(set(list_BSs) - set(busy_BSs))
            moderate_BSs = random.sample(list_BSs, num_moderate_BSs)
            list_BSs = list(set(list_BSs) - set(moderate_BSs))
            less_busy_BSs = list_BSs.copy()

            # Dividing in three groups
            busy_val = 0.6 * ts_resampled[t] / len(busy_BSs)
            moderate_val = 0.3 * ts_resampled[t] / len(moderate_BSs)
            low_val = 0.1 * ts_resampled[t] / len(less_busy_BSs)

            demand_vector[busy_BSs, row_idx, t] = busy_val
            demand_vector[moderate_BSs, row_idx, t] = moderate_val
            demand_vector[less_busy_BSs, row_idx, t] = low_val

    return demand_vector



def plot_lagrangian_bounds(zt_mu, z_master, iter_counter, mu_ub, start_index, end_index):
    x_values = [*range(1, iter_counter + 1)]
    print(len(zt_mu), len(z_master), len(x_values))
    fig, ax = plt.subplots()
    # z_opt = 421502.392571894 # Cost > 30 on on_sj and x_sj
    # z_opt = 381500.044657966 # Cost 1
    plt.plot(x_values, zt_mu, label='lower bound', color='g')
    plt.plot(x_values, z_master, label='upper bound', color='b')
    # plt.hlines(z_opt, x_values[0], x_values[-1], label='optimal', color='k')
    plt.title("Upper bound of mu = {}".format(mu_ub))
    plt.legend()
    plt.savefig('lagrangian-bounds-t{}-{}.png'.format(start_index, end_index))
    plt.cla()
    plt.close(fig)
    return


def run_greedy_heuristic(scenario, num_BSs, num_DCs, num_apps, start_time_index, end_time_index, metric):
    time_slots = [*range(start_time_index, end_time_index)]
    num_DCs_open_list = []
    energy_per_DC = []
    memory_per_DC = [[] for t in range(num_time_slots)]
    utilization_per_DC = [[] for t in range(num_time_slots)]
    throughput_per_DC = [[] for t in range(num_time_slots)]
    latency_per_app = [[] for t in range(num_time_slots)]
    greedy_sol_total_energy = 0
    prev_apps_loaded = np.array([[0 for j in range(num_DCs)] for s in range(num_apps)])
    apps_loaded_per_DC = [[[] for s in range(num_apps)] for _ in range(num_time_slots)]
    objective_value = 0.0
    prev_open_DCs = np.array([0 for _ in range(num_DCs)])
    time_start = time.time()
    total_objective_value_with_fixed_loading_cost = 0.0
    total_energy_cost = 0
    total_latency_cost = 0
    total_cloud_cost = 0
    for t in range(start_time_index, end_time_index):
        print("************\nTIME SLOT = {}".format(t))
        if metric == "latency":
            util_DCs, fraction_assigned, apps_loaded, DCs_open, num_DCs_open = greedy_solution_bfd_baseline_latency(
                scenario, t, prev_open_DCs, prev_apps_loaded)
        elif metric == "capacity":
            util_DCs, fraction_assigned, apps_loaded, DCs_open, num_DCs_open = greedy_solution_bfd_baseline_capacity(
                scenario, t, prev_open_DCs, prev_apps_loaded)
        num_DCs_open_list.append(num_DCs_open)
        apps_loaded_curr_timeslot = apps_loaded - prev_apps_loaded
        np.clip(apps_loaded_curr_timeslot, 0, 1, out=apps_loaded_curr_timeslot)
        prev_apps_loaded = apps_loaded.copy()
        

        greedy_solution = SolutionObject(scenario, None, num_DCs_open, fraction_assigned, DCs_open, apps_loaded,
                                        apps_loaded_curr_timeslot, None)
        energy, utilization_per_DC[t] = greedy_solution.calculate_energy_DCs(t)
        energy_per_DC.append(energy)
        greedy_sol_total_energy = greedy_sol_total_energy + np.sum(energy)

        latency_per_app[t] = greedy_solution.calculate_average_latency(t)
        memory_per_DC[t] = greedy_solution.calculate_memory_utilization(t)

        objective_fixed_loading_cost, energy_cost, latency_cost, cloud_cost = greedy_solution.calculate_objective_with_fixed_load_costs(
            t)
        total_objective_value_with_fixed_loading_cost = total_objective_value_with_fixed_loading_cost + \
                                                        objective_fixed_loading_cost
        total_energy_cost = total_energy_cost + energy_cost
        total_latency_cost = total_latency_cost + latency_cost
        total_cloud_cost = total_cloud_cost + cloud_cost

        for s in range(network_scenario.num_apps):
            loaded_DCs = np.nonzero(apps_loaded[s])
            loaded_list = [j for j in loaded_DCs[0]]
            apps_loaded_per_DC[t-start_time_index][s].extend(loaded_list)

        total_requests_DC = greedy_solution.calculate_total_requests_on_DCs(t)
        total_requests_cloud = greedy_solution.calculate_demand_on_cloud(t)

        demand_vector = np.array(network_scenario.demand)
        demand_sum_time_slot = np.sum(demand_vector[:, :, t], axis=(0, 1))
        print("Time slot = {}, demand = {}, assigned demand = {}".format(t,
                                                                         demand_sum_time_slot,
                                                                         sum(total_requests_DC) + total_requests_cloud))

        throughput_per_DC[t] = (np.array(total_requests_DC) / network_scenario.time_slot_in_seconds).tolist()
        print("Time slot = {}, demand assigned to cloud = {}, percent = {}".format(t,
                                                                                   total_requests_cloud,
                                                                                   total_requests_cloud * 100 / demand_sum_time_slot))
        print("Time slot = {}, throughput of DCs = {}".format(t, throughput_per_DC[t]))

        print("Time slot = {}, num DCs open = {}".format(t, num_DCs_open))
        print("Time slot = {}, Utilization of DCs = {}".format(t, utilization_per_DC[t]))
        print("Time slot = {}, Memory utilization of DCs = {}".format(t, memory_per_DC[t]))
        print("Time slot = {}, Objective value = {}, energy cost = {}, latency cost = {}, cloud cost = {}".format(t,
                                                                                                                  objective_fixed_loading_cost,
                                                                                                                  energy_cost,
                                                                                                                  latency_cost,
                                                                                                                  cloud_cost))
        greedy_solution.print_app_loaded_per_DC(t)
        # Set previously opened DCs for next iteration
        prev_open_DCs = np.array(DCs_open.copy())
    print("Time taken = {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start))))


    print("Latency per app per time slot = {}".format(latency_per_app))
    print("App_groups = {}".format(app_groups))
    print("Total energy consumed by all DCs in all time slots = {}".format(greedy_sol_total_energy))
    print("Throughput per DC per time slot = {}".format(throughput_per_DC))
    print("Utilization per DC per time slot = {}".format(utilization_per_DC))
    print("Objective components with fixed loading cost = {},{},{},{}".format(total_objective_value_with_fixed_loading_cost,
                                                      total_energy_cost,
                                                      total_latency_cost,
                                                      total_cloud_cost))
    return energy_per_DC, greedy_sol_total_energy


def solve_lagrangian(master_scenario, scenarios, num_BSs, num_DCs, num_apps, start_time_index, end_time_index,
                     with_benders, num_parallel_processes):
    # Create network_scenario objects for each time slot
    # with demand for that particular time slot
    num_time_slots = end_time_index - start_time_index
    time_slots = [*range(start_time_index, end_time_index)]


    # initialization of variables
    all_mus = []
    mu = np.array([[[0. for _ in range(num_time_slots)] for _ in range(num_DCs)] for _ in range(num_apps)])

    tolerance = 0.001  # stopping tolerance of 0.1 percent
    all_mus.append(mu)  # record the intermediate miu's
    iter_counter = 1
    all_sub_results = []
    time_start = time.time()
    iter_start_time = time.time()
    sub_results = run_model_parallel(scenarios, num_time_slots, mu.copy(), time_slots, iter_counter,
                                     with_benders, num_parallel_processes)
    all_sub_results.append(sub_results)
    zt_mu = [sum(sub_results[t + start_time_index][0] for t in range(num_time_slots))]
    z_master = []
    mu_ub = 100 

    while True:
        # solve the master problem
        print("Solving master problem")
        master_prob = LagrangianMaster(master_scenario, all_sub_results, num_time_slots, iter_counter, mu_ub)
        z_estimated, new_mu = master_prob.solve_master()
        new_mu = np.array(new_mu)
        all_mus.append(new_mu)  # keep track of all mu values
        z_master.append(z_estimated)

        improve = (z_estimated - zt_mu[-1]) / z_estimated
        print('Time: {} -- Iter time {} Iter {} -- Gap {} -- UB {} -- LB {}'.format(
            time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)),
            time.strftime('%H:%M:%S', time.gmtime(time.time() - iter_start_time)),
            iter_counter, improve, z_estimated, zt_mu[-1]))
        with open("lagrangian_{}_{}_mu.txt".format(start_time_index, end_time_index), "a") as mufile:
            mufile.write("Iter {} mu = {}\n".format(iter_counter, new_mu.tolist()))
        with open("lagrangian_{}_{}_subresults.txt".format(start_time_index, end_time_index), "a") as resultfile:
            resultfile.write("Iter {} subresults = {}\n".format(iter_counter, sub_results))

        if improve <= tolerance:
            print("\n##### Break as improvement {} below tolerance {} #####\n".format(improve, tolerance))
            print("Final Multiplier: {}, max = {}, min = {}".format(all_mus[-1], np.max(all_mus[-1]), np.min(all_mus[-1])))
            print("Lower Bound: {}".format(zt_mu[-1]))
            plot_lagrangian_bounds(zt_mu, z_master, iter_counter, mu_ub, start_time_index, end_time_index)
            return zt_mu[-1], all_sub_results[-1]
        elif time.time() - time_start > 18000:
            print("\n##### Break as time limit reached #####\n")
            plot_lagrangian_bounds(zt_mu, z_master, iter_counter, mu_ub, start_time_index, end_time_index)
            return zt_mu[-1], all_sub_results[-1]


        iter_counter = iter_counter + 1
        iter_start_time = time.time()
        sub_results = run_model_parallel(scenarios, num_time_slots, new_mu.copy(), time_slots, iter_counter,
                                         with_benders, num_parallel_processes)
        all_sub_results.append(sub_results)
        new_zt_mu = sum(sub_results[t + start_time_index][0] for t in range(num_time_slots))
        zt_mu.append(new_zt_mu)


def solve_cplex_one_step_ahead(network_scenario, num_BSs, num_DCs, num_apps, start_time_index, end_time_index):
    obj_value_list = []
    num_DCs_open_list = []
    w_tisj_sol_list = []
    z_tj_sol_list = []
    x_tsj_sol_list = []
    on_tsj_sol_list = []
    slack_ts_sol_list = []

    energy_per_DC = [[] for t in range(num_time_slots)]
    throughput_per_DC = [[] for t in range(num_time_slots)]
    utilization_per_DC = [[] for t in range(num_time_slots)]
    memory_per_DC = [[] for t in range(num_time_slots)]
    latency_per_app = [[] for t in range(num_time_slots)]
    apps_loaded_per_DC = [[[] for s in range(num_apps)] for _ in range(num_time_slots)]
    time_start = time.time()

    total_energy = 0
    total_objective_value = 0
    total_objective_value_with_fixed_loading_cost = 0
    total_energy_cost = 0
    total_latency_cost = 0
    total_cloud_cost = 0

    for t in range(start_time_index, end_time_index):
        print("************\nTIME SLOT = {}".format(t))
        if t == start_time_index:
            prev_xsj_value = [[0 for j in range(num_DCs)] for s in range(num_apps)]

        objective_value, num_DCs_open, w_tisj_solution, \
        z_tj_solution, x_tsj_solution, on_tsj_solution, slack_ts_solution = \
            solve_one_step_ahead(network_scenario, t, prev_xsj_value)

        obj_value_list.append(objective_value)
        num_DCs_open_list.append(num_DCs_open)
        w_tisj_sol_list.append(w_tisj_solution)
        z_tj_sol_list.append(z_tj_solution)
        x_tsj_sol_list.append(x_tsj_solution)
        prev_xsj_value = x_tsj_solution.copy()
        on_tsj_sol_list.append(on_tsj_solution)
        slack_ts_sol_list.append(slack_ts_solution)

        # Each instance of optimization problem will be solved by
        # passing the previous value of x_s,j,t
        cplex_solution = SolutionObject(network_scenario, objective_value, num_DCs_open,
                                       w_tisj_solution, z_tj_solution, x_tsj_solution,
                                       on_tsj_solution, slack_ts_solution)
        total_objective_value = total_objective_value + objective_value
        objective_fixed_loading_cost, energy_cost, latency_cost, cloud_cost = cplex_solution.calculate_objective_with_fixed_load_costs(
            t)
        total_objective_value_with_fixed_loading_cost = total_objective_value_with_fixed_loading_cost + \
                                                        objective_fixed_loading_cost
        total_energy_cost = total_energy_cost + energy_cost
        total_latency_cost = total_latency_cost + latency_cost
        total_cloud_cost = total_cloud_cost + cloud_cost

        energy_per_DC[t - start_time_index], utilization_per_DC[t-start_time_index] = cplex_solution.calculate_energy_DCs(t)
        total_energy = total_energy + np.sum(energy_per_DC[t - start_time_index])
        latency_per_app[t - start_time_index] = cplex_solution.calculate_average_latency(t)
        memory_per_DC[t - start_time_index] = cplex_solution.calculate_memory_utilization(t)

        for s in range(network_scenario.num_apps):
            loaded_DCs = np.nonzero(x_tsj_solution[s])
            loaded_list = [j for j in loaded_DCs[0]]
            apps_loaded_per_DC[t - start_time_index][s].extend(loaded_list)

        total_requests_DC = cplex_solution.calculate_total_requests_on_DCs(t)
        total_requests_cloud = cplex_solution.calculate_demand_on_cloud(t)

        demand_vector = np.array(network_scenario.demand)
        demand_sum_time_slot = np.sum(demand_vector[:,:,t], axis=(0, 1))
        print("Time slot = {}, demand = {}, assigned demand = {}".format(t,
                                                                         demand_sum_time_slot,
                                                                         sum(total_requests_DC) + total_requests_cloud))

        throughput_per_DC[t - start_time_index] = (np.array(total_requests_DC) / network_scenario.time_slot_in_seconds).tolist()
        print("Time slot = {}, demand assigned to cloud = {}, percent = {}".format(t,
                                                                                   total_requests_cloud,
                                                                                   total_requests_cloud*100/demand_sum_time_slot))
        print("Time slot = {}, throughput of DCs = {}".format(t, throughput_per_DC[t - start_time_index]))
        print("Time slot = {}, num DCs open = {}".format(t, num_DCs_open))
        print("Time slot = {}, Utilization of DCs = {}".format(t, utilization_per_DC[t - start_time_index]))
        print("Time slot = {}, Memory utilization of DCs = {}".format(t, memory_per_DC[t - start_time_index]))
        print("Time slot = {}, Objective value = {}, energy cost = {}, latency cost = {}, cloud cost = {}".format(t,
                                                                                                                  objective_fixed_loading_cost,
                                                                                                                  energy_cost,
                                                                                                                  latency_cost,
                                                                                                                  cloud_cost))
        cplex_solution.print_app_loaded_per_DC(t)
        cplex_solution.check_slack_values()

        # cplex_solution.print_app_loaded_per_DC()

    print("Time taken = {}".format(time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start))))
    print("Latency per app per time slot = {}".format(latency_per_app))
    print("Throughput per DC per time slot = {}".format(throughput_per_DC))
    print("Utilization per DC per time slot = {}".format(utilization_per_DC))
    print("Total objective value = {}".format(total_objective_value))
    print("Total energy consumed by all DCs in all time slots = {}".format(total_energy))
    print("Objective components with fixed loading cost = {},{},{},{}".format(total_objective_value_with_fixed_loading_cost,
                                                      total_energy_cost,
                                                      total_latency_cost,
                                                      total_cloud_cost))
    


def solve_cplex_complete(network_scenario, num_BSs, num_DCs, num_apps, start_time_index, end_time_index):
    start_index, objective_value, num_DCs_open, w_tisj_solution, \
    z_tj_solution, x_tsj_solution, on_tsj_solution, slack_ts_solution = run_cplex(network_scenario, start_time_index)

    latency_per_app = [[] for t in range(num_time_slots)]
    memory_per_DC = [[] for t in range(num_time_slots)]
    energy_per_DC = [[] for t in range(num_time_slots)]
    apps_loaded_per_DC = [[[] for s in range(num_apps)] for _ in range(num_time_slots)]
    total_objective_value_with_fixed_loading_cost = 0
    total_energy_cost = 0
    total_latency_cost = 0
    total_cloud_cost = 0
    for t in range(num_time_slots):
        curr_time_slot = t + start_time_index
        print("************\nTIME SLOT = {}".format(curr_time_slot))
        cplex_solution = SolutionObject(network_scenario, objective_value, num_DCs_open[t],
                                       w_tisj_solution[t], z_tj_solution[t], x_tsj_solution[t],
                                       on_tsj_solution[t], slack_ts_solution[t])


        energy_per_DC[t], utilization_per_DC = cplex_solution.calculate_energy_DCs(t)
        objective_fixed_loading_cost, energy_cost, latency_cost, cloud_cost = cplex_solution.calculate_objective_with_fixed_load_costs(
            t)
        total_objective_value_with_fixed_loading_cost = total_objective_value_with_fixed_loading_cost + \
                                                        objective_fixed_loading_cost
        total_energy_cost = total_energy_cost + energy_cost
        total_latency_cost = total_latency_cost + latency_cost
        total_cloud_cost = total_cloud_cost + cloud_cost

        for s in range(network_scenario.num_apps):
            loaded_DCs = np.nonzero(x_tsj_solution[t][s])
            loaded_list = [j for j in loaded_DCs[0]]
            apps_loaded_per_DC[t][s].extend(loaded_list)
        latency_per_app[t] = cplex_solution.calculate_average_latency(t)
        memory_per_DC[t] = cplex_solution.calculate_memory_utilization(t)

        print("Time slot = {}, num DCs open = {}".format(t, num_DCs_open))
        print("Time slot = {}, Utilization of DCs = {}".format(t, utilization_per_DC))
        print("Time slot = {}, Memory utilization of DCs = {}".format(t, memory_per_DC[t]))
        print("Time slot = {}, Objective value = {}, energy cost = {}, latency cost = {}, cloud cost = {}".format(t,
                                                                                                                  objective_fixed_loading_cost,
                                                                                                                  energy_cost,
                                                                                                                  latency_cost,
                                                                                                                  cloud_cost))
        cplex_solution.print_app_loaded_per_DC(t)
        cplex_solution.check_slack_values()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time_index', '-s', dest='start_time_index', type=int, required=True)
    parser.add_argument('--end_time_index', '-e', dest='end_time_index', type=int, required=True)
    parser.add_argument('--num_BSs', '-b', dest='num_BSs', type=int, required=False, default=20)
    # Data centers (DCs) are referred to as edge nodes (ENs) in the paper
    parser.add_argument('--num_DCs', '-d', dest='num_DCs', type=int, required=False, default=5) 
    parser.add_argument('--num_apps', '-a', dest='num_apps', type=int, required=False, default=20)
    parser.add_argument('--azure_day', '-azd', dest='azure_day', type=int, required=False, default=1)
    parser.add_argument('--method', '-m', dest='method', type=str, required=True)
    parser.add_argument('--lagrangian_num_parallel', dest='lagrangian_num_parallel', type=int)
    parser.add_argument('--lagrangian_with_benders', dest='with_benders', action='store_true')
    parser.add_argument('--lagrangian_without_benders', dest='with_benders', action='store_false')
    parser.add_argument('--use_azure_dataset', dest='use_azure_dataset', action='store_true')
    parser.add_argument('--azure_dataset_path', dest='azure_dataset_path', type=str)
    parser.add_argument('--min_cpu', dest='min_cpu', type=int, required=False)
    parser.add_argument('--max_cpu', dest='max_cpu', type=int, required=False)
    parser.add_argument('--min_demand', dest='min_demand', type=int, required=False)
    parser.add_argument('--max_demand', dest='max_demand', type=int, required=False)
    parser.add_argument('--seed', dest='seed', type=int, required=False, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.use_azure_dataset and args.azure_dataset_path is None:
        parser.error("--use_azure_dataset requires --azure_dataset_path")
    if args.method == "lagrangian" and args.lagrangian_num_parallel is None:
        parser.error("Running lagrangian method requires number parallel processes --lagrangian_num_parallel")

    time_slot_in_minutes = 15
    time_slot_in_seconds = time_slot_in_minutes * 60
    num_time_slots = 96
    day = args.azure_day
    
    if args.use_azure_dataset:
        # Top 25 by median function invocations
        hash_functions = ["5315be05fc3b21a3f483ed0759bce825764dcf8a762623a1d94ff63f9d9ce4cc",
                          "0ce67779eaa33056a996ccdeabeb3c04e48d41388d5fa734c30f505d7583b559",
                          "47658c9cff05caaffd85722b6f81dd163f680f03d4026474a098f966e2f528b3",
                          "9f7c42b1b5e58255e47691a7b28b08272a9e6a7db596871d300d5ed6a0363e35",
                          "b200945b3dd7c23a6e1c971016f448d15002495c7052015d02a6ffee8799cefc",
                          "68c45f54d6dd3861acc40bcebba5441c90e55a2a0228a07cc629e26e291ea3ca",
                          "b64fc6e2fbb22d3360090c85255445d161a7b01f716e17a9d8e3a4096a9960eb",
                          "ba854003e05348b105e9d92c219614a8093a82bc2b53aed8f42f8084c5bf00b8",
                          "8a3b5e4fd8e6d07528c9c135b7870bcb762579cfe7482293206f2eccd15ba1bb",
                          "2aec780b74edb50d77742f296089f7066ab78085a9414a96c53ca61513deda48",
                          "c1f661674fda24278bf9ad049cd6]9ba3638e43f07a778bdc121b75d87ffb4669",
                          "875652920ad86e74f898a9dfa1a0a6d75c94f7aa794d3591659111603c7ff872",
                          "a12d7ca0af06ae5a59eb1a04dd7d87491ed56a261d786f690c215afbfc158338",
                          "8c9ce0de516e04d52c78b54b7d51a5fbe406bdcc2e61d19d18646b064488947d",
                          "6c871093253858350d730f80cfbc5109aa2529d9cda37341989ea8854485663e",
                          "8097587729935b9b340fb933a512af18fa37233601aa4bd487e439a646359ab9",
                          "d2bf1b1520d605516698538c0183afec5c5317148815c4f990fb1996935250e0",
                          "02158978e403150c2befd8b4176c717c4230186b25f7de41a83e27a13f0dd4e4",
                          "740756f1353f6840c650d5d06043d13916a24213ec954993bdecfc97bf408364",
                          "1e96dba25531d7884b3c21f04badfdd784f86b943fe92058211f18ed087bf15b",
                          "b8fb8be49781176b16bc5c3dfd27c8ed67a4a1a135be766e5a9ded8057185087",
                          "101e17ae7801e4acefab9ca1d685221eeda35fd6edd3ee644028cfac0534d31e",
                          "299875d577dd3e0adba6d0f060f9b246c30a0415c5d8d8232ab03c3d7ba7d6d7",
                          "b9ffe67ce971ec2bf68ae5d92970dae21d1a611b230807b28be13b2ea8e8beeb",
                          "eddf4557fae85aae06da41619ddefacb898bb7d170891153c266148405eb791c",
                          "ff7e5bd1750ba364ae53323bc654c98f5576f8528af0eb8957c30b26034a605f"]


        function_trace_df = read_azure_trace_by_hash_function(day, hash_functions,
                                                              args.azure_dataset_path.rstrip("/") + "/")

        demand_vector = calculate_demand_vector_varying_busyness(function_trace_df, num_time_slots, args.num_apps,
                                                                 args.num_BSs, time_slot_in_minutes)

        app_groups = {"Video": [*range(0, 5)],
                      "Compute": [*range(5, 10)],
                      "AR": [*range(10, 15)],
                      "Health": [*range(15, 20)]
                      }
        resilience_per_app = {"Video": 1,
                              "Compute": 2,
                              "AR": 2,
                              "Health": 1}
        colors_per_app = {"Video": "g",
                          "Compute": "k",
                          "AR": "r",
                          "Health": "y"}
        latency_cloud = [112, 115, 114, 107, 117, 109, 101, 106, 115, 119, 114, 114, 107, 113, 103, 115, 100, 112,
                         116, 101]
        compute_capacities = [22000 * time_slot_in_seconds for j in range(args.num_DCs)]
        mem_capacities = [32000 for j in range(args.num_DCs)]
        latency_DC = [[28, 13, 36, 29, 11], [11, 37, 14, 36, 45], [10, 34, 25, 32, 13], [27, 48, 10, 35, 14],
                      [14, 40, 12, 37, 38], [36, 44, 45, 10, 10], [48, 26, 40, 10, 13], [28, 44, 11, 13, 36],
                      [13, 36, 36, 11, 48], [49, 43, 11, 38, 10], [35, 41, 10, 10, 46], [10, 47, 26, 13, 29],
                      [11, 44, 12, 38, 26], [32, 45, 38, 14, 13], [13, 37, 11, 49, 47], [49, 13, 10, 28, 35],
                      [44, 12, 46, 42, 11], [36, 28, 12, 13, 28], [46, 11, 38, 12, 46], [30, 14, 10, 25, 38]]

        network_scenario = NetworkScenario(time_slot_in_seconds, args.num_BSs, args.num_DCs, args.num_apps,
                                           demand_vector.tolist(), app_groups.copy(), args.start_time_index,
                                           latency_cloud, latency_DC, compute_capacities, mem_capacities,
                                           resilience_per_app)
    else:
        # Random number of requests per 15 min per BS and per app
        demand_vector = np.array([[[int(random.uniform(args.min_demand, args.max_demand)) for t in range(num_time_slots)]
                            for s in range(args.num_apps)] for i in range(args.num_BSs)])

        # Divide apps between groups randomly
        divide_apps_groups = np.array_split([*range(0,args.num_apps)], 4)
        app_groups = {"Video": divide_apps_groups[0].tolist(),
                      "Compute": divide_apps_groups[1].tolist(),
                      "AR": divide_apps_groups[2].tolist(),
                      "Health": divide_apps_groups[3].tolist()
                      }
        resilience_per_app = {"Video": 1,
                              "Compute": 1,
                              "AR": 1,
                              "Health": 1}
        colors_per_app = {"Video": "g",
                          "Compute": "k",
                          "AR": "r",
                          "Health": "y"}

        latency_cloud = [int(random.uniform(100, 120)) for i in range(args.num_BSs)]
        latency_DC = [[0 for j in range(args.num_DCs)] for i in range(args.num_BSs)]
        for i in range(args.num_BSs):
            total_DCs = [*range(0, args.num_DCs)]
            close_DCs = np.random.choice(total_DCs, 2, replace=False)
            for j in close_DCs:
                latency_DC[i][j] = int(random.uniform(10, 15))
            further_DCs = set(total_DCs).difference(set(close_DCs))
            for j in further_DCs:
                latency_DC[i][j] = int(random.uniform(25, 50))
        compute_capacities = [int(random.uniform(args.min_cpu, args.max_cpu)) * time_slot_in_seconds \
                              for j in range(args.num_DCs)]
        mem_capacities = [32000 for j in range(args.num_DCs)]
        network_scenario = NetworkScenario(time_slot_in_seconds, args.num_BSs, args.num_DCs, args.num_apps,
                                           demand_vector.tolist(), app_groups.copy(), args.start_time_index,
                                           latency_cloud, latency_DC, compute_capacities, mem_capacities,
                                           resilience_per_app)

    num_time_slots = args.end_time_index - args.start_time_index  # Number of time slots to be run

    if args.method == "lagrangian":
        # Create individual network scenarios to solve in parallel
        scenarios = []
        for time_slot in range(args.start_time_index, args.end_time_index):
            demand_sub_vector = demand_vector[:, :, time_slot:time_slot + 1].copy().tolist()
            scenarios.append(NetworkScenario(time_slot_in_seconds, args.num_BSs, args.num_DCs, args.num_apps,
                                             demand_sub_vector, app_groups.copy(), time_slot,
                                             latency_cloud, latency_DC, compute_capacities, mem_capacities,
                                             resilience_per_app))

        obj_value, lagrangian_solution = solve_lagrangian(network_scenario, scenarios, args.num_BSs, args.num_DCs,
                                                          args.num_apps, args.start_time_index, args.end_time_index,
                                                          args.with_benders, args.lagrangian_num_parallel)
    elif args.method == "cplex_complete":
        solve_cplex_complete(network_scenario, args.num_BSs, args.num_DCs, args.num_apps, args.start_time_index,
                             args.end_time_index)
    elif args.method == "cplex_one_step":
        solve_cplex_one_step_ahead(network_scenario, args.num_BSs, args.num_DCs, args.num_apps, args.start_time_index,
                                 args.end_time_index)
    elif args.method == "greedy_capacity":
        run_greedy_heuristic(network_scenario, args.num_BSs, args.num_DCs, args.num_apps, args.start_time_index,
                             args.end_time_index, "capacity")
    elif args.method == "greedy_latency":
        run_greedy_heuristic(network_scenario, args.num_BSs, args.num_DCs, args.num_apps, args.start_time_index,
                             args.end_time_index, "latency")
    else:
        print("Unknown method")