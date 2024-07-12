from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime
import time
import pickle
import os
import networkx as nx

import multiprocessing as mp
from functools import partial

from utils.aggregate_statistics import aggregate_statistics
from utils.configure_input_data_graph_airsea import add_input_params_to_db_structures

from run_sim_model_obj_complexbs import run_sim_model


def calculate_objectives(pool, lock, solutions, dataset, dec_var_db, **kwargs):
    """Calculate objective as the distance between the ground truth data and the simulated data using the
    Manhattan metrics for each actor in the supply chain. We normalize the distance for each actor.
    The final objective is the sum of the distance of each actor.

    return
    -------
    Array of the objective value of each solution in the shape (batchsize, )"""

    # Change the first parameter into a graph
    values_to_round = solutions[:, 0]
    selected_dec_var = {round(value): dec_var_db[round(value)] for value in values_to_round}

    del dec_var_db

    time.sleep(1)
    results_all = pool.map(partial(run_sim_model, df_truemodel=dataset, dec_var=selected_dec_var,
                                   lock=lock), solutions)

    del selected_dec_var

    # pyribs automatically maximizes, so we use a negative obj value (distance should be small)
    obj_values_all = [-v[0] for v in results_all]
    behaviour_kpi = [v[1] for v in results_all]

    del results_all, dataset, solutions

    return np.array(obj_values_all), np.array(behaviour_kpi)


def calculate_improvements(prev, new_arch):
    """ Calculates the number of improvements per iteration. An improvement is defined as a change in the objective value
    for a container. If the objective value of a container is different (better) from the previous iteration, it is considered an
    improvement. The number of improvements is counted for all containers. """

    if prev is None:
        return 0
    else:
        new = {i.index: i for i in [elite for elite in new_arch]}

        improv_or_not = [(value.objective != new[key].objective) for key, value in prev.items() if key in new]
        num_improv = improv_or_not.count(True)

        del new, improv_or_not

        return num_improv


def reformat_solution_for_sim_cmaes(solutions, new_bounds):
    """Reformat the solutions to the format that the simulation model can use. This is necessary because the solutions
    are in the range of -1 to 1, while the simulation model uses different ranges for the decision variables."""

    full_sol = []
    for i in solutions:
        full_i = []
        all_index_sol = iter(range(len(i)))  # Convert to iterator for efficiency
        for n in new_bounds:
            if isinstance(n, tuple):
                new_value = next(all_index_sol)  # Use next() to get the next index
                new_value_reformat = 0.5 * (i[new_value] + 1) * (n[1] - n[0]) + n[0]
                full_i.append(new_value_reformat)

                del new_value_reformat
            else:
                full_i.append(n)
        full_sol.append(full_i)

        del full_i
        del all_index_sol

    return np.array(full_sol)


if __name__ == '__main__':
    # Configure the ground truth and the problem
    df_in = pd.read_csv(
        r"../../complex_stylized_supply_chain_model_generator/data/"
        "20240418_QD_ComplexSimModelGraph_GT_CNHK_USA_EventTimeSeries_ManufacturingTime1.5_Runtime364.csv")
    df_truemodel = aggregate_statistics(df_in)
    del df_in

    # # Import dictionary with decision variables of the graphs
    # with open(r"../complex_stylized_supply_chain_model_generator/data/location_model_hpc_40000_cnhk_usa_airsea.pkl", "rb") as file:
    #     dec_var_db = pickle.load(file)

    with open(r"../dec_var_qd_density_40000_cnhk_usa.pkl", "rb") as file:
        dec_var = pickle.load(file)

    start_date = datetime.now().strftime("%Y%m%d_%H%M%S")  # date.today().strftime("%Y%m%d")
    folder_name = start_date + "_QD_0_CMAES_Import"
    os.makedirs(r"./results/" + folder_name)

    # Set Emitter (is the algorithm)
    # these are the values of the parameters
    bounds_parameters = np.array([(0, max(dec_var.keys())), (10, 10), (1.5, 1.5),
                                  (0.5, 0.5), (0, 0), (0.5, 0.5),
                                  (0.5, 0.5), (0.5, 0.5), (1, 1),
                                  (0.5, 0.5), (1, 1), (1, 1), (0, 0),
                                  (1, 1), (1, 1), (0, 0),
                                  (0.1, 1), (0.5, 5), (0.5, 0.5),
                                  (0.5, 0.5), (0.5, 0.5), (0, 0),
                                  (0.5, 0.5), (0.5, 0.5), (0, 0),
                                  (0.1, 1), (0.3, 2), (0.5, 0.5),
                                  (0.5, 0.5), (0.5, 0.5), (1, 1),
                                  (0.2, 0.2), (0.1, 0.1)])
    # new_bounds = [(lower if lower == upper else (lower, upper)) for lower, upper in bounds_parameters]
    # del bounds_parameters
    # make smaller
    # tuple_bounds = [item for item in new_bounds if isinstance(item, tuple)]
    # nr_input_params = len(tuple_bounds)
    nr_input_params = len(bounds_parameters)
    initial_solution = np.zeros((nr_input_params,))
    initial_bounds = np.array([(-1, 1) for _ in range(nr_input_params)])

    # Set Archive
    # Solution dimensions is the total number of parameters of the input space
    archive = GridArchive(solution_dim=nr_input_params,
                          dims=[10, 10],
                          ranges=[(0.02, 0.07), (250, 1250)],
                          # this is the archive for the behaviour space so ranges for behaviour space
                          )
    # choose zeros as initial solutions since we do not have any prior knowledge of the model
    # choose only zeros makes a lot of infeasible solutions (even within the bounds of the dimensions)
    # at initial stepsize to be 1.0 so that it is sample from a standard isotropic Gaussian
    # therefore we choose the mean of the bounds with a sample size of 0.5
    seeds = [1, 5, 6, 10, 11, 15]

    with mp.Pool(processes=30) as pool:  # mp.cpu_count() #, maxtasksperchild=1
        for s in seeds:
            os.makedirs(fr"./results/{folder_name}/seed_{s}")
            emitters = [
                EvolutionStrategyEmitter(
                    archive,
                    x0=initial_solution.flatten(),
                    sigma0=0.5,
                    batch_size=90,
                    bounds=initial_bounds,
                    seed=s,
                    # es=CMAEvolutionStrategySolConstraints
                )
            ]

            # Set optimizer/scheduler
            scheduler = Scheduler(archive, emitters)

            # Optimize
            n_iterations = 120  # 125

            prev_archive = None
            itr_improvements = {}

            start_time = time.time()
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), f" Start optimization for MO 0% with seed {s}")

            lock = mp.Manager().Lock()

            for itr in range(n_iterations):
                solutions = scheduler.ask()
                # print(solutions)
                # reformat it for the simulation model -- solutions
                # use same function for processing the results
                solutions_reformat = 0.5 * (solutions + 1) * (
                            bounds_parameters[:, 1] - bounds_parameters[:, 0]) + bounds_parameters[:, 0]
                # solutions_reformat = reformat_solution_for_sim_cmaes(solutions, new_bounds)

                # Optimize the distance between the ground truth and the simulated data
                # BSc are two KPIs of the simulation model. We can use kwargs to easily change it (not implemented right now)
                objectives, bcs = calculate_objectives(pool, lock, solutions_reformat, df_truemodel, dec_var)

                # BCs: last 2 coordinates of the solutions, so the parameters on the length of the links.
                # bcs = solutions[:, 2:]

                scheduler.tell(objectives, bcs)
                del objectives, bcs, solutions, solutions_reformat

                # calculate convergence
                num_improvements = calculate_improvements(prev_archive, archive)
                itr_improvements[itr] = {"improvements_quality": num_improvements,
                                         "coverage_diversity": archive.stats.coverage,
                                         "num_elites_diversity": archive.stats.num_elites}
                prev_archive = {i.index: i for i in [elite for elite in archive]}

                if itr % 10 == 0:
                    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " Stats for ", itr, ":", archive.stats)

                    # Save intermediate results
                    results_intermediate = dict(archive=archive, emitter=emitters, optimizer=scheduler,
                                                convergence=itr_improvements)
                    file_name = f"./results/{folder_name}/seed_{s}" + "/Intermediate_Results_CMAME_Iterations=" + str(
                        itr) + ".pkl"

                    with open(file_name, 'wb') as output:
                        pickle.dump(results_intermediate, output)

                    del results_intermediate

            # Optimization ended
            end_time = time.time() - start_time
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " End optimization with run time of",
                  round(end_time, 1),
                  "seconds")

            # Save results
            results_opt = dict(archive=archive, emitter=emitters, optimizer=scheduler, convergence=itr_improvements,
                               sim_time=end_time)
            file_name = f"./results/{folder_name}/seed_{s}" + "/_Results_CMAME_Iterations=" + str(n_iterations) + ".pkl"

            with open(file_name, 'wb') as output:
                pickle.dump(results_opt, output)

            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " Results are pickled and saved")

            # Create 2D Heatmap
            grid_archive_heatmap(archive, cmap=plt.get_cmap('YlGnBu').reversed())
            # Save figure
            name = start_date + "_Heatmap_CMAME_Iterations=" + str(n_iterations)  # date.today().strftime("%Y%m%d")
            plt.savefig(f"./results/{folder_name}/seed_{s}/" + str(name) + ".png")
            # plt.show()