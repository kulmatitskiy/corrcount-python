from __future__ import division
from table import *
from scipy.stats import poisson
from numpy import random
from model_component import ModelComponent
from segment import SegmentComponent
from zero_inflater import ZeroInflaterComponent
import pandas as pd
import matplotlib.pyplot as plt

class PoissonComponentWithLatentClasses(ModelComponent):
    conditionals_axes = [segments_axis, observations_axis, categories_axis]

    def __init__(self, count_data, num_segments=2, inflated_zeros=False):
        super(PoissonComponentWithLatentClasses, self).__init__(count_data, num_segments)

        self.inflated_zeros = inflated_zeros
        if self.inflated_zeros:
            self.conditionals_axes = [segments_axis, didBuy_axis, observations_axis, categories_axis]
        else:
            self.conditionals_axes = [segments_axis, observations_axis, categories_axis]

        self.lambdas = self.generate_initial_estimates()

        self.segment_probs = None
        self.segment_names = None

    def generate_initial_estimates(self):
        # lambdas is S-by-K matrix
        k = self.count_data.num_categories
        lambdas = self.global_means_by_category().repeat(self.num_segments).reshape([self.num_segments, k], order='F')
        lambdas += random.ranf(size=lambdas.size).reshape(lambdas.shape)
        return lambdas

    def global_means_by_category(self):
        counts = self.count_data.arr
        if self.inflated_zeros:
            tmp = counts > 0
            return counts.sum(axis=0).astype('float64')/tmp.sum(axis=0)  # todo: use freqs
        else:
            return counts.mean(axis=0)  # todo: use freqs

    def params(self):
        return self.lambdas

    def poisson_prob_arr(self, means):
        # count_data must be N-by-K matrix of counts per observation N, category K
        # means must be S-by-K matrix of poisson means per segment C, category K
        # returns S-by-N-by-K probability table of observed count_data (given each segment)
        means = np.transpose(means.reshape([1] + list(means.shape)), [1,0,2]) # (1,S,K)
        counts = self.count_data.arr
        counts = counts.reshape([1] + list(counts.shape))
        return poisson(means).pmf(counts)

    def get_conditionals(self):
        poisson_part = self.poisson_prob_arr(self.lambdas)
        if self.inflated_zeros:
            almost_zero_means = np.zeros([self.num_segments, self.count_data.num_categories]) + 1e-2
            null_part = self.poisson_prob_arr(almost_zero_means)
            both_parts = np.transpose(np.array([poisson_part, null_part]), [1, 0, 2, 3])
            return Table(both_parts, axes=self.conditionals_axes)
        else:
            return Table(poisson_part, axes=self.conditionals_axes)

    def use_posteriors(self, posteriors, obtained_from):
        super(PoissonComponentWithLatentClasses, self).use_posteriors(posteriors, obtained_from)

        if isinstance(obtained_from, ZeroInflaterComponent) and self.inflated_zeros:
            self.deflate_posteriors = obtained_from.view_of_deflate_posteriors(posteriors) # S-by-N-by-K

        elif isinstance(obtained_from, SegmentComponent):
            s = self.num_segments
            n, _ = self.count_data.arr.shape
            if not self.inflated_zeros:
                c_mat = posteriors.get_arr([segments_axis, observations_axis])  # S-by-N
                c_sums = posteriors.get_arr([segments_axis], apply_func=np.sum)  # S-vec
                SC = np.dot(c_mat, self.count_data.arr) # SC is S-by-K
                S = len(c_sums)
                self.lambdas = SC / c_sums.reshape(S, 1)

            else:
                def_posteriors = np.transpose(self.deflate_posteriors, [2, 0, 1])  # K-S-N
                segm_posteriors = posteriors.get_arr([segments_axis, observations_axis]).reshape([1, s, n])
                c_arr = def_posteriors * segm_posteriors  # K-S-N
                # c_sums = c_arr.sum(axis=2) # sum over N
                sums = np.einsum('ijk->ji', c_arr)
                prods = np.einsum('ijj->ij', np.einsum('ijk,kl->jil', c_arr, self.count_data.arr)) # S-K
                # prods = np.einsum('ijk,kl->jil', c_arr, self.count_data.arr) # S-K
                self.lambdas = prods / sums

    def use_params_for_fitted_info(self, params, from_component):
        if isinstance(from_component, ZeroInflaterComponent):
            self.deflate_probs = from_component.params()
        if isinstance(from_component, SegmentComponent):
            self.segment_names = from_component.segment_names()
            pass # todo

    def fitted_info(self):
        cols = ["Exp Counts in %s" % c for c in self.count_data.category_names]
        means = self.lambdas
        if self.inflated_zeros:
            means *= self.deflate_probs
        return pd.DataFrame(np.round(means, decimals=1), index=self.segment_names, columns=cols)

    def use_params_for_fitted_plots(self, params, from_component):
        if isinstance(from_component, ZeroInflaterComponent):
            self.deflate_probs = from_component.params()
        if isinstance(from_component, SegmentComponent):
            self.segment_probs = from_component.params()

    def fitted_plots(self):
        n = self.count_data.number_of_observations()

        m = 16
        count_range = np.array(range(m))
        nbs = np.repeat(np.array(count_range), self.num_segments).reshape([self.num_segments, m], order='F')  # S-by-m
        segment_probs = self.segment_probs.reshape(self.num_segments, 1)  # make sure shape is S-by-1

        figs = list()
        for cat in range(self.count_data.num_categories):
            counts_observed = [(self.count_data.arr[:, cat] == count).sum() for count in count_range]

            poisson_means = self.lambdas[:,cat].reshape(self.num_segments, 1)  # S-by-1
            poisson_pmf = poisson(poisson_means).pmf(nbs)  # S-by-m
            if self.inflated_zeros:
                deflate_probs = self.deflate_probs[:, cat].reshape(self.num_segments, 1)
                poisson_pmf *= deflate_probs
                poisson_pmf += (1-deflate_probs) * poisson(1e-2).pmf(nbs)
            counts_predicted = (poisson_pmf * segment_probs).sum(axis=0) * n

            fig = plt.figure()
            plt.plot(count_range, counts_predicted, '--D', label="Fitted", zorder=4)
            plt.bar(count_range-0.45, counts_observed, width=0.9, color='lightgray', label="Observed", linewidth=1, edgecolor="gray",zorder=3)
            plt.grid(axis='y', which='major', color=(0.1, 0.1, 0.1), linestyle=':',zorder=0)
            plt.xlabel("Count of %s" % self.count_data.category_names[cat], fontsize=16)
            plt.ylabel("Number of observations", fontsize=16)
            plt.xticks(range(16), fontsize=13)
            plt.tick_params('both', length=0, width=1, which='major')
            plt.title("Aggregate Comparison at NumSegments=%d" % self.num_segments,fontsize=18)
            plt.legend(fontsize=16)
            plt.gca().set_axis_bgcolor((0.98, 0.98, 0.99))
            plt.xlim(-1,15.9)
            figs.append(fig)