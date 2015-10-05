from table import *
from zero_inflater import ZeroInflaterComponent
from poisson import PoissonComponentWithLatentClasses
import logging
import numpy as np

class Network(object):
    def __init__(self, component_classes, data, num_segments, dealing_with_zero_probs='force'):
        self.num_segments = num_segments
        self.count_data = CountData(data)
        self.dealing_with_zero_probs = dealing_with_zero_probs

        components = list()
        inflated_zeros = False
        for comp_class in component_classes:
            if comp_class is ZeroInflaterComponent:
                inflated_zeros = True

            if comp_class is PoissonComponentWithLatentClasses and inflated_zeros:
                components.append(comp_class(self.count_data, num_segments, True))
            else:
                components.append(comp_class(self.count_data, num_segments))
        self.components = components

    def check_zeros(self, in_table, affected_tables, check_sums_for_axis=None):
        zero_probs_mask = (in_table.arr == 0)

        if zero_probs_mask.any():
            if self.dealing_with_zero_probs == 'force':
                in_table.arr[zero_probs_mask] == 1e-4

            elif self.dealing_with_zero_probs == 'exclude':
                if check_sums_for_axis is not None:
                    raise NotImplementedError() # todo: implement removal of just those obs where sum(prob) is zero
                else:
                    _, bad_obs_inds = in_table.get_observations_inds(zero_probs_mask)

                for table in set([in_table] + affected_tables):
                    if table is not None:
                        table.remove(bad_obs_inds, observations_axis)
            else:
                raise Exception()  # unknown way of dealing with zero probabilities

    def run_once(self):
        reversed_components = list(reversed(self.components))
        prob_table = None
        expll_table = None
        posteriors_table = None
        for i, component in enumerate(reversed_components):

            # 1. Use current component's get_conditionals to update probability table
            conditionals = component.get_conditionals()

            self.check_zeros(in_table=conditionals, affected_tables=[self.count_data, expll_table],
                             check_sums_for_axis=component.main_axis)

            if prob_table is None:
                prob_table = ProbTable(self.count_data, conditionals)
            else:
                prob_table.update(conditionals)  # mutates prob_table

            posteriors = None
            if component.is_latent:
                # 2a. Use probability table to compute posteriors table, if appropriate
                posteriors = prob_table.get_scaled(axes_to_sum_over=[component.main_axis])
                # 2b. Call subsequent components to use new posteriors to update their estimates
                for subseq_component in reversed_components[:(i+1)]:
                    subseq_component.use_posteriors(posteriors, obtained_from=component)

            # 3. expected log-likelihood given data
            if expll_table is None:
                expll_table = ExpLLTable(self.count_data, conditionals)
            else:
                expll_table.update(conditionals, posteriors)  # mutates expll_table

        return expll_table.get_ll()

    def run(self, threshold=1e-12, max_iterations=250, num_attempts=1, store_traces=False):
        e_ll_traces = list()
        e_ll_best = -np.inf
        for attempt in range(num_attempts):
            e_ll = -np.inf
            e_ll_trace = list()
            for i in range(max_iterations):
                prev_e_ll = e_ll
                e_ll = self.run_once()  # <-- run once

                if store_traces:
                    e_ll_trace.append(e_ll)

                logging.info("Iteration %d trying to improve %f." % (i, e_ll))

                if e_ll < prev_e_ll:
                    logging.warning("Expected ll value DECREASED at iteration %d." % i)

                if np.isnan(e_ll).any():
                    logging.error("Got NaN for likelihood! Abort attempt.")
                    break

                if abs(e_ll - prev_e_ll) <= threshold:
                    break

            e_ll_traces.append(e_ll_trace)

            if e_ll > e_ll_best:
                e_ll_best = e_ll
                for component in self.components:
                    component.save_params()

        self.e_ll_traces = e_ll_traces
        self.e_ll_best = e_ll_best

    def get_excluded_observations(self):
        return self.count_data.excluded_observations

    def get_total_number_of_params(self):
        return sum([c.params().size for c in self.components])

    def fitted_component_info(self, comp_ind):
        component_wanted = self.components[comp_ind]
        for i in range(comp_ind+1):
            prev_component = self.components[i]
            component_wanted.use_params_for_fitted_info(prev_component.params(), from_component=prev_component)
        return component_wanted.fitted_info()

    def fitted_component_plots(self, comp_ind):
        component_wanted = self.components[comp_ind]
        for i in range(comp_ind+1):
            prev_component = self.components[i]
            component_wanted.use_params_for_fitted_plots(prev_component.params(), from_component=prev_component)
        return component_wanted.fitted_plots()



