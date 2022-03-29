This repository contains code for the optimization and evaluation in the article [Energy-Efficient Service Placement for Latency-Sensitive Applications in Edge Computing](https://ieeexplore.ieee.org/document/9743551) by [Gopika Premsankar](https://gpremsan.github.io) and [Bissan Ghaddar](https://www.ivey.uwo.ca/faculty/directory/bissan-ghaddar/) published in the IEEE Internet of Things Journal (2022).

## 1. Setting up the environment

Our code uses IBM ILOG CPLEX Python API (**version 20.1** of CPLEX Optimization Studio). Follow IBM's [download](https://www.ibm.com/support/pages/how-do-i-download-cplex-optimization-studio) and [installation instructions](https://www.ibm.com/docs/en/icos/20.1.0?topic=2010-installing-cplex-optimization-studio) to set up CPLEX on your system. 

Set up a Python virtual environment with the following steps. The code was tested using **Python 3.8.10**. If you have multiple Python versions on your system, we recommend using **python3.8** as follows.

```commandline
python3.8 -m venv venv
source venv/bin/activate
```

Assuming CPLEX 20.1 is installed on your computer, set up CPLEX Python API in the virtual environment by first navigating to where CPLEX is installed. 
E.g. On a Linux-based system, the location is: `/opt/ibm/ILOG/CPLEX_Studio201/cplex/python/3.8/x86-64_linux/` in which case
`$CPLEX_HOME` is `/opt/ibm/ILOG/CPLEX_Studio201/cplex`, `$VERSION` is `3.8` and `$PLATFORM` is `x86-64_linux`. 

Accordingly, run the following steps to install CPLEX Python API to your virtual environment. `$PATH_TO_THIS_PROJECT` is the path where this Git repository was cloned.

```commandline
cd $CPLEX_HOME/python/$VERSION/$PLATFORM
python3.8 setup.py install --home $PATH_TO_THIS_PROJECT/venv/lib/python3.8/site-packages
```

Return to this project, activate the virtual environment (if not already done), and install other dependencies.

```commandline
cd $PATH_TO_THIS_PROJECT
source venv/bin/activate
pip install -r requirements.txt
```

Note that you must set the Python path every time you run the project, otherwise an error `ModuleNotFoundError: No module named 'cplex'` will be thrown.

```commandline
export PYTHONPATH=$PYTHONPATH:$CPLEX_HOME/python/$VERSION/$PLATFORM
```

## 2. Running the code

The script `evaluate_networks.py` is used to run the code. There are several mandatory arguments to be passed to set up the scenario, including the number of base stations, number of edge nodes, number of applications, etc.

You can see all the required arguments through:

```commandline
python evaluate_networks.py --help
```

An example run for a small network using the one-step ahead heuristic is:

```commandline
python evaluate_networks.py -b 3 -d 5 -a 5 --min_cpu 10000 --max_cpu 20000 --min_demand 0 --max_demand 300 -m cplex_one_step -s 0 -e 3
```

which indicates that a network with 3 base stations, 5 edge nodes and 5 applications are set up. The minimum and maximum 
CPU capacities of the edge nodes are drawn from Uniform(10000,20000). The demand for each application from each base 
station is drawn from Uniform(0, 300). Once the demand vector is generated, the heuristic one-step ahead is used to solve for the placement and scheduling for the first three time periods (`-s 0` indicates to use the demand vector starting with time period 0 and `-e 3` indicates ending with time period 3). The `method` argument takes `cplex_one_step`, `greedy_capacity` and `greedy_latency` that correspond to the One-step ahead, Greedy-by-capacity and Greedy-by-latency heuristics described in the paper.

The following is an example command to obtain the Lagrangian solution:

```commandline
python evaluate_networks.py -b 10 -d 5 -a 5 --min_cpu 10000 --max_cpu 20000 --min_demand 0 --max_demand 300 -m lagrangian --lagrangian_num_parallel 1 --lagrangian_with_benders -s 0 -e 3
```

This will run until the gap between the upper bound and lower bound is less than 0.1 percent. At the end of the run, three output files are generated, which plot the intermediate results. Set the `lagrangian_num_parallel` value according to the number of cores that you have on your system; this determines how many subproblems are run in parallel and the number depends on your system configuration.

## 3. Running the code using the dataset from Microsoft Azure

Download the traces (2019) for the paper "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider" (ATC'19) from [here](https://github.com/Azure/AzurePublicDataset). Extract the downloaded tar file into a directory, the path to which `$PATH_TO_DOWNLOADED_DATASET$` is required when running the code to read the downloaded files. The files that are required for this project are `invocations_per_function_md.anon.d{}.csv`. 

An example run using the downloaded traces is:

```commandline
python evaluate_networks.py -s 0 -e 96 -b 20 -d 5 -a 20 -m cplex_one_step -azd 1 --use_azure_dataset --azure_dataset_path <PATH_TO_DOWNLOADED_DATASET>
```
This first generates a demand vector using the data from the first day (`azd 1`), and solves for the placement and scheduling decisions for the 96 time periods. The network comprises of 20 base stations, 5 edge nodes and 20 applications.


## 4. Interpreting the logs

The code prints detailed log information to the standard output. The logs contain the number of edge nodes open in each time slot, the utilization of the ENs, applications that are loaded, and the total cost of the solution. Some sample scripts are also provided in the `scripts/` folder that can be used to interpret the logs. To use the scripts, first redirect the logs from running `evaluate_networks.py` to a text file and provide the path to the output files to each script file. Ensure that the saved log (output) files follow the naming convention indicated in each script (see the `filename` that is read in the script). `generate_plots.py` were used to generate Figures 3 and 4 in the article, `plot_azure_demand.py` to generate Figure 2 and `plot_utilization.py` to generate Figure 5. Finally, the `SolutionObject` in `class_definitions.py` provides additional functions that can be called in `evaluate_networks.py` to analyze intermediate output objects.

## 5. Citation


Please cite the following paper in your publication if our code helps your research. 

     @article{premsankar2022energy,
      title={Energy-Efficient Service Placement for Latency-Sensitive Applications in Edge Computing},
      author={Premsankar, Gopika and Ghaddar, Bissan},
      journal={IEEE Internet of Things Journal}, 
      year={2022},
      doi={10.1109/JIOT.2022.3162581}
    }

## 6. Acknowledgements

Bissan Ghaddar was supported by the David G. Burgoyne Faculty Fellowship and an NSERC Discovery Grant 2017-04185. Gopika Premsankar was
supported by an Academy of Finland grant 338854 and a postdoc pool grant from the Finnish Cultural Foundation.

