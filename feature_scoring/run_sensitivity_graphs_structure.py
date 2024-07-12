"""
Created on: 10-10-2022 18:56

@author: IvS
"""

import pandas as pd
import sys
import numpy as np
import pickle
from collections import Counter
from RunnerObject_qd import (
    ComplexSupplyChainSimModelQD
)
from utilities_simmodel import calculate_manhattan_distance
from utils.aggregate_statistics import aggregate_statistics


from ema_workbench import Model, RealParameter, MultiprocessingEvaluator, Constant, ScalarOutcome, \
    ema_logging, save_results, CategoricalParameter, IntegerParameter

ema_logging.log_to_stderr(ema_logging.INFO)


def run_sim_model(df_truemodel, graphs, graph_idx, interarrival_time, manufacturing_time,
                  wholesales_consolidator_time_min, wholesales_consolidator_time_mode,
                  wholesales_consolidator_time_max,
                  wholesales_consolidator_pickuptime_min, wholesales_consolidator_pickuptime_mode,
                  wholesales_consolidator_pickuptime_max,
                  wholesales_consolidator_prob_on_container,
                  export_port_sea_time_min, export_port_sea_time_mode, export_port_sea_time_max,
                  transit_port_sea_time_min, transit_port_sea_time_mode, transit_port_sea_time_max,
                  import_wait_on_steck_time_sea_min, import_wait_on_steck_time_sea_max,
                  import_prob_extracting_sea,
                  export_port_air_time_min, export_port_air_time_mode, export_port_air_time_max,
                  transit_port_air_time_min, transit_port_air_time_mode, transit_port_air_time_max,
                  import_wait_on_steck_time_air_min, import_wait_on_steck_time_air_max,
                  import_prob_extracting_air,
                  wholesales_distributor_time_min, wholesales_distributor_time_mode,
                  wholesales_distributor_time_max,
                  large_retailer_time,
                  small_retailer_time):
    graph = graphs[graph_idx]
    del graphs

    sol = [graph, interarrival_time, manufacturing_time,
           wholesales_consolidator_time_min, wholesales_consolidator_time_mode, wholesales_consolidator_time_max,
           wholesales_consolidator_pickuptime_min, wholesales_consolidator_pickuptime_mode,
           wholesales_consolidator_pickuptime_max,
           wholesales_consolidator_prob_on_container,
           export_port_sea_time_min, export_port_sea_time_mode, export_port_sea_time_max,
           transit_port_sea_time_min, transit_port_sea_time_mode, transit_port_sea_time_max,
           import_wait_on_steck_time_sea_min, import_wait_on_steck_time_sea_max,
           import_prob_extracting_sea,
           export_port_air_time_min, export_port_air_time_mode, export_port_air_time_max,
           transit_port_air_time_min, transit_port_air_time_mode, transit_port_air_time_max,
           import_wait_on_steck_time_air_min, import_wait_on_steck_time_air_max,
           import_prob_extracting_air,
           wholesales_distributor_time_min, wholesales_distributor_time_mode, wholesales_distributor_time_max,
           large_retailer_time,
           small_retailer_time]

    results_sim_model, kpis_simmodel = ComplexSupplyChainSimModelQD.run(parameters=sol)

    transport_cost = kpis_simmodel["Transport_Cost"]

    del sol, kpis_simmodel

    obj_dist = []
    for colname, colval in results_sim_model.items():
        true_col = df_truemodel[colname]
        dist = calculate_manhattan_distance(true_col, colval)

        # Normalize distance
        min_obj = true_col["p5"]
        max_obj = true_col["p95"]
        if min_obj == max_obj:
            dist = min(dist, 1) if dist > 0 else 0
        else:
            dist = (dist - min_obj) / (max_obj - min_obj)

        obj_dist.append(dist)
    total_distance = sum(obj_dist)

    del results_sim_model

    return total_distance, transport_cost


if __name__ == '__main__':
    # Configure the ground truth and the problem
    df_in = pd.read_csv(
        r"../../complex_stylized_supply_chain_model_generator/data/"
        "20231208_FINAL_ComplexSimModelGraph_GT_CNHK_USA_EventTimeSeries_ManufacturingTime1.5_Runtime364.csv")
    df_truemodel = aggregate_statistics(df_in)

    # Import dictionary with decision variables of the graphs
    with open(r"../dec_var_qd_density_40000_cnhk_usa.pkl", "rb") as file:
        dec_var = pickle.load(file)

    graphs = tuple([v["graph"] for k, v in dec_var.items()])
    del dec_var

    # Define the model
    model = Model('SupplyChainModel', function=run_sim_model)

    # Specify uncertainties -- not working! for some reason not taking all the graphs as index

    model.uncertainties = [IntegerParameter("graph_idx", 0, len(graphs)-1),
                           ]

    # Specify constants
    model.constants = [Constant("df_truemodel", df_truemodel),
                       Constant('graphs', graphs),
                           Constant('interarrival_time', 10),
                           Constant('manufacturing_time', 1.5),
                           Constant('wholesales_consolidator_time_min', 0.5),
                           Constant('wholesales_consolidator_time_mode', 0.5),
                           Constant('wholesales_consolidator_time_max', 1),
                           Constant('wholesales_consolidator_pickuptime_min', 0.5),
                           Constant('wholesales_consolidator_pickuptime_mode', 1),
                           Constant('wholesales_consolidator_pickuptime_max', 2),
                           Constant('wholesales_consolidator_prob_on_container', 0.5),
                           Constant('export_port_sea_time_min', 1),
                           Constant('export_port_sea_time_mode', 2),
                           Constant('export_port_sea_time_max', 2),
                           Constant("transit_port_sea_time_min", 1),
                           Constant("transit_port_sea_time_mode", 2),
                           Constant("transit_port_sea_time_max", 2),
                           Constant("import_wait_on_steck_time_sea_min", 0.5),
                           Constant("import_wait_on_steck_time_sea_max", 3),
                           Constant("import_prob_extracting_sea", 0.5),
                           Constant("export_port_air_time_min", 0.5),
                           Constant("export_port_air_time_mode", 1),
                           Constant("export_port_air_time_max", 1),
                           Constant("transit_port_air_time_min", 0.5),
                           Constant("transit_port_air_time_mode", 1),
                           Constant("transit_port_air_time_max", 1),
                           Constant("import_wait_on_steck_time_air_min", 0.5),
                           Constant("import_wait_on_steck_time_air_max", 1),
                           Constant("import_prob_extracting_air", 0.5),
                           Constant("wholesales_distributor_time_min", 0.5),
                           Constant("wholesales_distributor_time_mode", 1),
                           Constant("wholesales_distributor_time_max", 2),
                           Constant("large_retailer_time", 0.2),
                           Constant("small_retailer_time", 0.1)]

    # Specify outcomes
    model.outcomes = [ScalarOutcome("total_distance"),
                      ScalarOutcome("transport_cost")]

    with MultiprocessingEvaluator(model, n_processes=47) as evaluator: #-1
        results = evaluator.perform_experiments(scenarios=100)

    # results[0]["graph_idx"] = idx
    # results[0]["#nodes"] = dec_var[idx]["nodes"]
    # results[0]["#edges"] = dec_var[idx]["edges"]
    # results[0]["density"] = dec_var[idx]["density"]
    # node_types = [data.get('node_type') for _, data in graph.nodes(data=True)]
    # type_counts = Counter(node_types)
    # for node_type, count in type_counts.items():
    #     results[0][node_type] = count

    # Save the results
    save_results(results, f"Results_EMA_Graph_Graph_Structure.tar.gz")
