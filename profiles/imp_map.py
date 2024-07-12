from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

from cma_es_with_sol_constraints import CMAEvolutionStrategySolConstraints

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

    del results_all

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


if __name__ == '__main__':
    # Configure the ground truth and the problem
    df_in = pd.read_csv(
        r"../../complex_stylized_supply_chain_model_generator/data/"
        "20240322_QD_ComplexSimModelGraph_GT_CNHK_USA_EventTimeSeries_ManufacturingTime1.5_Runtime364.csv")
    df_truemodel = aggregate_statistics(df_in)

    # # Import dictionary with decision variables of the graphs
    # with open(r"../complex_stylized_supply_chain_model_generator/data/location_model_hpc_40000_cnhk_usa_airsea.pkl", "rb") as file:
    #     dec_var_db = pickle.load(file)

    with open(r"../dec_var_qd_density_40000_cnhk_usa.pkl", "rb") as file:
        dec_var = pickle.load(file)

    start_date = datetime.now().strftime("%Y%m%d_%H%M%S")  # date.today().strftime("%Y%m%d")
    folder_name = start_date + "_QD_0_MAP_Import"
    os.makedirs(r"./results/" + folder_name)

    # Set Emitter (is the algorithm)
    # these are the values of the parameters
    bounds_parameters = np.array([(0, max(dec_var.keys())), (10, 10), (1.5, 1.5),
                                  (0.5, 0.5), (0.5, 0.5), (1, 1),
                                  (0.5, 0.5), (1, 1), (2, 2),
                                  (0.5, 0.5), (1, 1), (2, 2), (2, 2),
                                  (1, 1), (2, 2), (2, 2),
                                  (0.1, 1), (0.5, 5), (0.5, 0.5),
                                  (0.5, 0.5), (1, 1), (1, 1),
                                  (0.5, 0.5), (1, 1), (1, 1),
                                  (0.1, 1), (0.3, 2), (0.5, 0.5),
                                  (0.5, 0.5), (1, 1), (2, 2),
                                  (0.2, 0.2), (0.1, 0.1)])
    nr_input_params = len(bounds_parameters)
    initial_gaussian_center = np.array([np.mean(bounds) for bounds in bounds_parameters])
    initial_gaussian_std = np.array([np.mean(bounds) / 2 if (np.mean(bounds) != bounds[0])
                                                            and (np.mean(bounds) != bounds[1]) else 0 for bounds in
                                     bounds_parameters])

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
    seeds = range(5)
    emitters = [
        GaussianEmitter(
            archive,
            x0=initial_gaussian_center.flatten(),
            sigma=initial_gaussian_std.flatten(),
            batch_size=96,  # 50
            bounds=bounds_parameters,
            seed=s,
        ) for s in seeds
    ]

    # Set optimizer/scheduler
    scheduler = Scheduler(archive, emitters)

    # Optimize
    n_iterations = 200

    prev_archive = None
    itr_improvements = {}

    start_time = time.time()
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " Start optimization")

    lock = mp.Manager().Lock()
    with mp.Pool(processes=48) as pool:  # mp.cpu_count()
        for itr in range(n_iterations):
            solutions = scheduler.ask()
            # print(solutions)

            # Optimize the distance between the ground truth and the simulated data
            # BSc are two KPIs of the simulation model. We can use kwargs to easily change it (not implemented right now)
            objectives, bcs = calculate_objectives(pool, lock, solutions, df_truemodel, dec_var)

            # BCs: last 2 coordinates of the solutions, so the parameters on the length of the links.
            # bcs = solutions[:, 2:]

            scheduler.tell(objectives, bcs)

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
                file_name = "./results/" + folder_name + "/Intermediate_Results_MAP_Iterations=" + str(
                    itr) + ".pkl"

                with open(file_name, 'wb') as output:
                    pickle.dump(results_intermediate, output)

        # Optimization ended
        end_time = time.time() - start_time
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " End optimization with run time of", round(end_time, 1),
              "seconds")

        # Save results
        results_opt = dict(archive=archive, emitter=emitters, optimizer=scheduler, convergence=itr_improvements,
                           sim_time=end_time)
        file_name = "./results/" + folder_name + "/_Results_CMAME_Iterations=" + str(n_iterations) + ".pkl"

        with open(file_name, 'wb') as output:
            pickle.dump(results_opt, output)

        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), " Results are pickled and saved")

        # Create 2D Heatmap
        grid_archive_heatmap(archive, cmap=plt.get_cmap('YlGnBu').reversed())
        # Save figure
        name = start_date + "_Heatmap_MAP_Iterations=" + str(n_iterations)  # date.today().strftime("%Y%m%d")
        plt.savefig("./results/" + folder_name + "/" + str(name) + ".png")
        # plt.show()
