import re
import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(utilization_vals, filename):
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Helvetica'})
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    for method_name in method_dict.keys():
        util_values = utilization_vals[method_name].copy()
        util_values = sorted(util_values)

        y_vals = np.arange(len(util_values))/float(len(util_values)-1)

        plt.plot(util_values, y_vals,
                 label=labels[method_name],
                 color=colors[method_name])

    plt.yticks(ticks=np.arange(0, 1.1, 0.1))
    plt.grid(alpha=0.7, linestyle=':')
    plt.legend()
    plt.ylabel("Cumulative probability")
    plt.xlabel("Utilization (%)")
    fig.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.cla()
    plt.close(fig)


def get_utilization_values():
    utilization_vals = {}

    for key in method_dict.keys():
        for day_num in range(1, 8):
            filename = DIR + "day{day_num}/{method}_logs_day{day_num}_azure_with_throughput.txt".format(
                        method=method_dict[key],
                        day_num=day_num)
            with open(filename, "r") as f:
                lines = f.readlines()
            if key == "cplex_onestep":
                line = lines[-4]
            elif key == "greedy_bfd_capacity":
                line = lines[-2]
            elif key == "greedy_bfd_latency":
                line = lines[-2]
            m = re.match("Utilization per DC per time slot = (.*)",
                         line)

            if m:
                all_util_values = np.array(eval(m.group(1)))
                all_util_values = all_util_values * 100
                all_util_values = all_util_values[all_util_values != 0] # Remove DCs which are not open
                all_util_values = all_util_values.flatten().tolist()

            else:
                print("Objective values not found in the logs {}".format(filename))
                print(line)

            if day_num == 1:
                utilization_vals[key] = all_util_values.copy()
            else:
                utilization_vals[key].extend(all_util_values)
    #print(utilization_vals)
    return utilization_vals




if __name__ == "__main__":
    increased_latency_threshold = False
    DIR = ""
    if len(DIR) == 0:
        print("Update the path to the results files before running script")
        exit()

    if increased_latency_threshold:
        max_latency_per_app = {"Video": 80,  # 70
                               "Compute": 50,
                               "AR": 50,
                               "Health": 70}
    else:
        max_latency_per_app = {"Video": 50,  # 70
                               "Compute": 20,
                               "AR": 20,
                               "Health": 40}

    method_dict = {
        "cplex_onestep": "cplex_one_step",
        "greedy_bfd_capacity": "greedy_capacity",
        "greedy_bfd_latency": "greedy_latency"
    }
    labels = {
        "cplex_onestep": "One-step ahead",
        "greedy_bfd_capacity": "Greedy-by-capacity",
        "greedy_bfd_latency": "Greedy-by-latency"
    }
    colors = {
        "cplex_onestep": "#1b9e77",
        "greedy_bfd_capacity": "#d95f02",
        "greedy_bfd_latency": "#7570b3"
    }

    utilization_vals = get_utilization_values()
    plot_cdf(utilization_vals.copy(), "plot-cdf-utilization-7days.png")