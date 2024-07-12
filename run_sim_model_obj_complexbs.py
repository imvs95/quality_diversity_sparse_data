"""
Created on: 10-10-2022 18:56

@author: IvS
"""

import time
from RunnerObject_qd import (
    ComplexSupplyChainSimModelQD
)
from utilities_simmodel import calculate_manhattan_distance


def run_sim_model(sol, df_truemodel, dec_var, lock):
    """ This function runs the simulation model with the given parameters and returns the distance between the
     simulated data and the ground truth data, and the behavior space. The distance is calculated using the Manhattan distance.

    The behavior space of this function is the density of the graph network, and the transport cost."""
    sol = sol.astype(object)
    nr_sol = sol[0]
    sol[0] = dec_var[round(nr_sol)]["graph"]

    results_sim_model, kpis_simmodel = ComplexSupplyChainSimModelQD.run(parameters=sol)

    behaviour_space_sol = [dec_var[round(nr_sol)]["density"], kpis_simmodel["Transport_Cost"]]

    del sol, kpis_simmodel, dec_var

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

    return total_distance, behaviour_space_sol
