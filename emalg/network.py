from table import *
import logging
import numpy as np

class Network(object):
    def __init__(self, component_classes, count_data, num_segments):
        self.num_segments = num_segments
        self.count_data = count_data
        self.components = [C(count_data=count_data, num_segments=num_segments) for C in component_classes]

    def check_zeros(self, in_table, affected_tables):
        zero_probs_mask = (in_table.arr == 0)
        if zero_probs_mask.any():
            _, bad_obs_inds = in_table.get_observations_inds(zero_probs_mask)
            for table in set([in_table] + affected_tables):
                if table is not None:
                    table.remove(bad_obs_inds, observations_axis)

    def run_once(self):
        reversed_components = list(reversed(self.components))
        prob_table = None
        expll_table = None
        posteriors_table = None
        for i, component in enumerate(reversed_components):

            # 1. Use current component's get_conditionals to update probability table
            conditionals = component.get_conditionals()

            self.check_zeros(in_table=conditionals, affected_tables=[self.count_data, expll_table])

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

    def run(self, threshold=1e-8, max_iterations = 200):
        e_ll = -np.inf
        e_ll_trace = []
        for i in range(max_iterations):
            prev_e_ll = e_ll
            e_ll = self.run_once()
            e_ll_trace.append(e_ll)
            if e_ll < prev_e_ll:
                logging.warning("Expected ll value DECREASED at iteration %d." % i)
            if abs(e_ll - prev_e_ll) <= threshold:
                break
        print(e_ll_trace)


