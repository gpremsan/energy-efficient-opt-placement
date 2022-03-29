import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import FuncFormatter

def get_time_taken_cplex(increased_latency_threshold):
    time_taken_all_days = []
    for day_num in range(1, 8):
        if increased_latency_threshold:
            filename = DIR + "day{day_num}/increased_latency_threshold/{method}_logs_day{day_num}_azure.txt".format(
                method=method_dict["cplex_onestep"],
                day_num=day_num)
        else:
            filename = DIR + "day{day_num}/{method}_logs_day{day_num}_azure.txt".format(
                method=method_dict["cplex_onestep"],
                day_num=day_num)
        with open(filename, "r") as f:
            lines = f.readlines()
        line = lines[-6]
        m = re.match("Time taken = (.*):(.*):(.*)", line)
        if m:
            num_hours = int(m.group(1))
            num_minutes = int(m.group(2))
            num_seconds = int(m.group(3))
            time_taken = num_hours*60*60 + num_minutes*60 + num_seconds
            time_taken_all_days.append(time_taken)
        else:
            print("Error. Time taken not found in log file {}".format(filename))
    return time_taken_all_days

def get_objective_values(increased_latency_threshold):
    plot_data_objective = {}
    plot_data_energy = {}

    for key in method_dict.keys():
        for day_num in range(1, 8):
            if increased_latency_threshold:
                filename = DIR + "day{day_num}/increased_latency_threshold/{method}_logs_day{day_num}_azure.txt".format(method=method_dict[key],
                                                                                        day_num=day_num)
            else:
                filename = DIR + "day{day_num}/{method}_logs_day{day_num}_azure.txt".format(
                            method=method_dict[key],
                            day_num=day_num)
            with open(filename, "r") as f:
                lines = f.readlines()
            if key == "cplex_onestep":
                line = lines[-2]
            elif key == "greedy_bfd_capacity":
                line = lines[-1]
            elif key == "greedy_bfd_latency":
                line = lines[-1]
            m = re.match("Objective components with fixed loading cost = (.*),(.*),(.*),(.*)",
                         line)

            if m:
                total_objective = float(m.group(1))
                total_energy_cost = float(m.group(2))
                total_latency_cost = float(m.group(3))
                total_cloud_cost = float(m.group(4))
            else:
                print("Objective values not found in the logs {}".format(filename))
                print(line)

            if day_num == 1:
                plot_data_objective[key] = [total_objective]
                plot_data_energy[key] = [total_energy_cost]
            else:
                plot_data_objective[key].append(total_objective)
                plot_data_energy[key].append(total_energy_cost)
    return plot_data_objective.copy(), plot_data_energy.copy()


def get_latency_values(method_dict, increased_latency_threshold):
    latency_vals = {}
    for key in method_dict.keys():
        latency_vals[key] = []
    #latency_vals = []  # num rows = num time slots, num_columns = num_apps
    for key in method_dict.keys():
        for day_num in range(1, 8):
            if increased_latency_threshold:
                filename = DIR + "day{day_num}/increased_latency_threshold/{method}_logs_day{day_num}_azure.txt".format(method=method_dict[key],
                                                                                            day_num=day_num)
            else:
                filename = DIR + "day{day_num}/{method}_logs_day{day_num}_azure.txt".format(
                    method=method_dict[key],
                    day_num=day_num)
            with open(filename, "r") as f:
                lines = f.readlines()
            if key == "cplex_onestep":
                line = lines[-5]
            elif key == "greedy_bfd_capacity":
                line = lines[-4]
            elif key == "greedy_bfd_latency":
                line = lines[-4]
            m = re.match("Latency per app per time slot = (.*)", line)

            if m:
                latency_per_day = eval(m.group(1))
                # print(len(latency_per_day), len(latency_per_day[0]))
                latency_vals[key].extend(latency_per_day)
    return latency_vals.copy()


def get_num_DCs_open(method_dict, increased_latency_threshold):
    plot_data_num_DCs_open = {}
    for key in method_dict.keys():
        plot_data_num_DCs_open[key] = [0] * 7 * 96
        for day_num in range(1, 8):
            if increased_latency_threshold:
                filename = DIR + "day{day_num}/increased_latency_threshold/{method}_logs_day{day_num}_azure.txt".format(
                    method=method_dict[key],
                    day_num=day_num)
            else:
                filename = DIR + "day{day_num}/{method}_logs_day{day_num}_azure.txt".format(
                                method=method_dict[key],
                                day_num=day_num)
            with open(filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                m = re.match("Time slot = (\d+), num DCs open = (\d+)", line)
                if m:
                    time_slot = int(m.group(1))
                    plot_data_num_DCs_open[key][((day_num-1) * 96) + time_slot] = int(m.group(2))
    return plot_data_num_DCs_open

def plot_latency_per_group(latencies, group_name, app_indices, max_latency, filename):
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Helvetica'})
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    line_plots = []
    for method_name in method_dict.keys():
        all_latencies = np.array(latencies[method_name])
        group_latencies = all_latencies[:, app_indices].copy()

        avg_latency = np.average(group_latencies, axis=1)
        max_observed_latency = np.max(group_latencies, axis=1)
        min_observed_latency = np.min(group_latencies, axis=1)

        time_slots = [*range(0, len(avg_latency))]
        p1, = plt.plot(time_slots, avg_latency, label=labels[method_name], color=colors[method_name])
        # plt.fill_between(time_slots, min_observed_latency, max_observed_latency, alpha=0.3, color=colors[method_name])
        line_plots.append(p1)

    #plt.legend(handles=line_plots, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(prop={'size': 11})
    plt.hlines(max_latency, time_slots[0], time_slots[-1], linestyle='dashed', color="black",
               linewidth=2, label=group_name)

    ax.set_ylim(ymin=0)
    plt.ylabel("Average latency (ms)")
    num_days = len(avg_latency) // 96
    ax.set_xticks([i * 96 for i in range(0, num_days)])
    ax.set_xticklabels(["  Day {}".format(i + 1) for i in range(0, num_days)])
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('left')
    fig.tight_layout()

    #plt.show()
    plt.savefig(filename, dpi=600)
    plt.cla()
    plt.close(fig)


def plot_latency_all_groups(latency_per_app, filename):
    latency = np.transpose(np.array(latency_per_app))
    fig, ax = plt.subplots()
    time_slots = [*range(0, len(latency[0]))]
    latency_per_group = {}
    line_plots = []
    max_observed_latency = {}
    min_observed_latency = {}
    for group_name, app_indices in app_groups.items():
        latency_per_group[group_name] = np.average(latency[app_indices, :], axis=0)
        max_observed_latency[group_name] = np.max(latency[app_indices, :], axis=0)
        min_observed_latency[group_name] = np.min(latency[app_indices, :], axis=0)
        # Plot min / max per group?
        p1, = plt.plot(time_slots, latency_per_group[group_name], label=group_name)
        #plt.fill_between(time_slots, min_observed_latency[group_name],
        #                 max_observed_latency[group_name], alpha=0.3, color='tab:gray')
        line_plots.append(p1)

    for app_group, max_latency in max_latency_per_app.items():
        plt.hlines(max_latency, time_slots[0], time_slots[-1], linestyle=':', label=app_group)
        plt.text(1, max_latency + 0.5, app_group, ha='left', va='center')
    #plt.title("Latency per application group")
    plt.legend(handles=line_plots, bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend(handles=line_plots, loc='upper left')
    ax.set_ylim(ymin=0)
    plt.ylabel("Average latency (ms)")
    num_days = len(latency[0])//96
    ax.set_xticks([i*96 for i in range(0,num_days)])
    ax.set_xticklabels(["Day {}".format(i+1) for i in range(0,num_days)])
    fig.tight_layout()

    plt.show()
    #plt.savefig(filename)
    #plt.cla()
    #plt.close(fig)


def plot_num_DCs_open(data_num_DCs_open, method_dict, filename):
    fig, ax = plt.subplots()
    time_slots = [*range(0, 7*96)]
    line_plots = []
    for method_name in method_dict.keys():
        p1,  = plt.plot(time_slots, data_num_DCs_open[method_name], label=labels[method_name], color=colors[method_name])
        line_plots.append(p1)
    plt.ylabel("Number of active edge nodes")
    ax.set_xticks([i * 96 for i in range(0, 7)])
    ax.set_xticklabels(["Day {}".format(i + 1) for i in range(0, 7)])
    plt.legend()
    fig.tight_layout()

    plt.show()

def SI(x, pos):
    if x == 0:
        return x
    bins = [1000000000000.0, 1000000000.0, 1000000.0, 1000.0, 1, 0.001, 0.000001, 0.000000001]
    abbrevs = ['T', 'G', 'M', 'k', '', 'm', 'u', 'n']
    label = x
    for i in range(len(bins)):
        if abs(x) >= bins[i]:
            label = '{1:.{0}f}'.format(1, x / bins[i]) + abbrevs[i]
            break

    return label


def plot_bar_chart(data, legend_labels, ylabel, filename):
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'font.sans-serif': 'Helvetica'})
    plt.rcParams.update({'font.size': 14})
    x_labels = ["Day {}".format(i) for i in range(1, 8)]
    x = np.arange(len(x_labels))  # the label locations
    width = 0.8 # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width / 3, data["cplex_onestep"], width/3, label=legend_labels["cplex_onestep"], color=colors["cplex_onestep"])
    ax.bar(x, data["greedy_bfd_capacity"], width/3, label=legend_labels["greedy_bfd_capacity"], color=colors["greedy_bfd_capacity"])
    ax.bar(x + width / 3, data["greedy_bfd_latency"], width/3, label=legend_labels["greedy_bfd_latency"], color=colors["greedy_bfd_latency"])

    if ylabel == "Energy costs":
        ax.set_ylim(ymax=790000)
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.gca().yaxis.set_major_formatter(FuncFormatter(SI))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.cla()
    plt.close(fig)

def print_latex_table_values(obj_values, time_vals_cplex):

    # Day 1, Greedy_latency, Greedy_cap, One step obj, one step time
    for day in range(1,8):
        idx = day-1
        print("{} & {:.2f} & {:.2f} & {:.2f} & {} \\\\".format(day, obj_values["greedy_bfd_latency"][idx], obj_values["greedy_bfd_capacity"][idx],
                                 obj_values["cplex_onestep"][idx], time_vals_cplex[idx]))

def calculate_latency_violations(latency_vals_method, max_latency_per_app, app_groups):
    latency_vals_method = np.transpose(np.array(latency_vals_method))
    total_violations = 0
    for app_group in app_groups.keys():
        app_indices = app_groups[app_group]
        max_latency = max_latency_per_app[app_group]+1
        avg_latencies = np.mean(latency_vals_method[app_indices], axis=0)
        violations = (avg_latencies>max_latency).sum()
        total_violations = total_violations + violations
        print(app_group, violations)
    print("Total number of violations = {}, per day = {}".format(total_violations, total_violations/7))


if __name__ == "__main__":
    increased_latency_threshold = False
    DIR = ""
    if len(DIR) == 0:
        print("Update the path to the results files before running script")
        exit()

    app_groups = {"Video": [*range(0, 5)],
                  "Compute": [*range(5, 10)],
                  "AR": [*range(10, 15)],
                  "Health": [*range(15, 20)]
                  }
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
        "cplex_onestep": "cplex_onestep_withoutbenders",
        "greedy_bfd_capacity": "greedy",
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

    num_DCs_open = get_num_DCs_open(method_dict, increased_latency_threshold)
    #plot_num_DCs_open(num_DCs_open, method_dict.copy(), None)
    latency_vals = get_latency_values(method_dict.copy(), increased_latency_threshold)
    objective_vals, energy_vals = get_objective_values(increased_latency_threshold)
    #
    time_taken_vals = get_time_taken_cplex(increased_latency_threshold)

    calculate_latency_violations(latency_vals["cplex_onestep"], max_latency_per_app, app_groups)
    calculate_latency_violations(latency_vals["greedy_bfd_capacity"], max_latency_per_app, app_groups)

    # print_latex_table_values(objective_vals, time_taken_vals)
    # print_latex_table_values(energy_vals, time_taken_vals)

    for group_name in ["Compute", "Video"]:
        if increased_latency_threshold:
            plot_latency_per_group(latency_vals.copy(), group_name,
                                   app_groups[group_name], max_latency_per_app[group_name],
                                   "plot-apps{}-latency-allmethods-increasedlatency.png".format(group_name))
        else:
            plot_latency_per_group(latency_vals.copy(), group_name,
                                   app_groups[group_name], max_latency_per_app[group_name],
                                   "plot-apps{}-latency-allmethods.png".format(group_name))

    if increased_latency_threshold:
        plot_bar_chart(energy_vals.copy(), labels, "Energy costs", "plot-barchart-energy-increasedlatency.png")
    else:
        plot_bar_chart(energy_vals.copy(), labels, "Energy costs", "plot-barchart-energy.png")

