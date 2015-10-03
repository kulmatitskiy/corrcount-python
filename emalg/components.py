from base import *
from numpy import random


class Component(object):
    prob_table_axes = list()

    def __init__(self, num_segments=2, count_data=None):
        raise NotImplementedError()

    def prob_table(self):
        raise NotImplementedError()

    def prob_table_in_shape(self, axes):
        my_names = self.prob_table_dim_names
        prob_table = self.prob_table()

        permutation = [my_names.index(ax.name) for ax in axes if ax.name in my_names]
        prob_table = np.transpose(prob_table, permutation)

        apply_shape = [size if ax.name in self.prob_table_dim_names else 1 for ax, size in zip(axes, arr.shape)]
        prob_table = prob_table.reshape(apply_shape)

        return prob_table

    def use(self, conditionals, component):
        pass


class XSegmentComponent(Component):
    prob_table_axes = [observations_axis, segments_axis]

    def __init__(self, count_data, num_segments=2):
        self.num_segments = num_segments
        self.count_data = count_data

        tmp = 0.8 * random.ranf(size=num_segments) + 0.1
        self.seg_probs = tmp / tmp.sum()

        #self.seg_probs = np.ones(num_segments, dtype='float64') / num_segments
        # TODO: init params using data (k-means?)

    def params(self):
        return self.seg_probs

    def prob_table(self):
        n, _ = self.count_data.arr.shape
        return self.seg_probs.repeat(n).reshape(n, self.num_segments, order='F')

    def use(self, conditionals, component):
        if component is self:
            posteriors = conditionals.get_normalized_conditionals(axes_to_sum_over=[segments_axis])
            n, _ = self.count_data.arr.shape
            self.seg_probs = posteriors.get_arr([segments_axis], apply_func=np.sum) / n



class XPoissonComponentWithLatentClasses(Component):
    prob_table_axes = [segments_axis, observations_axis, categories_axis]

    def __init__(self, count_data, num_segments=2):
        self.num_segments = num_segments
        self.count_data = count_data

        # lambdas is S-by-K matrix
        K = count_data.num_categories
        self.lambdas = self.initial_estimates().repeat(num_segments).reshape([num_segments, K], order='F')
        self.lambdas += random.ranf(size=self.lambdas.size).reshape(self.lambdas.shape)

    def initial_estimates(self):
        return self.count_data.arr.mean(axis=0) # todo: use freqs

    def params(self):
        return self.lambdas

    def poisson_prob_table(self, means):
        # count_data must be N-by-K matrix of counts per observation N, category K
        # means must be S-by-K matrix of poisson means per segment C, category K
        # returns S-by-N-by-K probability table of observed count_data (given each segment)
        means = np.transpose(means.reshape([1] + list(means.shape)), [1,0,2]) # (1,S,K)
        counts = self.count_data.arr
        counts = counts.reshape([1] + list(counts.shape))
        return poisson(means).pmf(counts)

    def prob_table(self):
        return self.poisson_prob_table(self.lambdas)

    def use(self, conditionals, component):
        if isinstance(component, XSegmentComponent):
            segm_posteriors = conditionals.get_normalized_conditionals(axes_to_sum_over=[segments_axis])
            segm_posteriors_mat = segm_posteriors.get_arr([segments_axis, observations_axis])  # S-by-N
            segm_posteriors_sums = segm_posteriors.get_arr([segments_axis], apply_func=np.sum) # S-vec
            #segm_posteriors_sums = segm_posteriors.get_sums(segments_axis) # S-vec
            S = len(segm_posteriors_sums)

            SC = np.dot(segm_posteriors_mat, self.count_data.arr) # SC is S-by-K
            self.lambdas = SC / segm_posteriors_sums.reshape(S, 1)


