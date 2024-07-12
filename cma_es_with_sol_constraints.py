"""
Created on: 20-11-2023 14:55

@author: IvS
"""

from ribs.emitters.opt._cma_es import CMAEvolutionStrategy
from threadpoolctl import threadpool_limits
from ribs._utils import readonly

import numpy as np


class CMAEvolutionStrategySolConstraints(CMAEvolutionStrategy):
    """This class is an extension of the CMA-ES optimization class of pyribs.
    For our specific problem, we need to find feasible solutions for distributions. Therefore, extra solution
    constraints are added to transform and check sol."""

    def __init__(self, sigma0, solution_dim, batch_size=None, seed=None, dtype=np.float64):
        super().__init__(sigma0,
                         solution_dim,
                         batch_size,
                         seed,
                         dtype)
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
            batch_size (int): batch size of the sample. Defaults to
                ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)
        self.cov.update_eigensystem(self.current_eval, self.lazy_gap_evals)
        transform_mat = self.cov.eigenbasis * np.sqrt(self.cov.eigenvalues)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        while len(remaining_indices) > 0:
            unscaled_params = self._rng.normal(
                0.0,
                self.sigma,
                (len(remaining_indices), self.solution_dim),
            ).astype(self.dtype)
            new_solutions, out_of_bounds = self._transform_and_check_sol(
                unscaled_params, transform_mat, self.mean, lower_bounds,
                upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Added for my own constraints -- IvS
            new_ofb = np.empty_like(out_of_bounds)
            for i, sol in enumerate(new_solutions):
                updated_ofb = CMAEvolutionStrategySolConstraints.check_all_distributions_of_solution(sol,
                                                                                                     out_of_bounds[i, :])
                new_ofb[i] = updated_ofb

            out_of_bounds = new_ofb

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

        return readonly(self._solutions)


    @staticmethod
    def check_all_distributions_of_solution(sol, out_of_bounds_sol):
        """This functions determines which parameters belong to which distribution. It is case specific."""
        # Set up all the distributions
        wholesales_distribution_time = (sol[3], sol[4], sol[5])
        wholesales_consolidator_pickuptime = (sol[6], sol[7], sol[8])
        export_port_sea_time = (sol[10], sol[11], sol[12])
        transit_port_sea_time = (sol[13], sol[14], sol[15])
        import_wait_on_steck_time_sea = (sol[16], sol[17])
        export_port_air_time = (sol[19], sol[20], sol[21])
        transit_port_air_time = (sol[22], sol[23], sol[24])
        import_wait_on_steck_time_air = (sol[25], sol[26])
        wholesales_distributor_time = (sol[28], sol[29], sol[30])

        list_distributions = [wholesales_distribution_time, wholesales_consolidator_pickuptime,
                              export_port_sea_time, transit_port_sea_time,
                              import_wait_on_steck_time_sea, export_port_air_time,
                              transit_port_air_time, import_wait_on_steck_time_air,
                              wholesales_distributor_time]

        # Check whether it is feasible
        check_distr = [item for distr in list_distributions for item in
                       CMAEvolutionStrategySolConstraints.check_feasible_params_distribution(distr)]

       # Update out of bounds
        new_out_of_bounds_sol = out_of_bounds_sol.copy()
        params_idx = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17,
                      19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30]
        for list_nr, idx in enumerate(params_idx):
            if new_out_of_bounds_sol[idx]:
                # this means it is already out of bound - don't change a thing
                pass
            elif check_distr[list_nr]:
                # this means that parameter of solution is feasible
                pass
            elif not check_distr[list_nr]:
                # this means that distribution is not feasible, so parameter also not
                new_out_of_bounds_sol[idx] = True

        return new_out_of_bounds_sol

    @staticmethod
    def check_feasible_params_distribution(distr: tuple):
        """ To check the feasibility of the parameters of a distribution
        which some parameters are. This is the coupling of QD with simulation.
        Input is a tuple of the parameters. Output is a Boolean - if True, the distribution is feasible,
        if False, the distribution is not."""
        if len(distr) == 2:
            # for Uniform
            feasible = distr[0] < distr[1]
        elif len(distr) == 3:
            # for Triangular
            feasible = distr[0] < distr[1] < distr[2]
        return [feasible] * len(distr)
