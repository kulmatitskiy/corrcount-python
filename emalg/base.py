from __future__ import division
import numpy as np
from scipy.stats import poisson


# Misc

def remove_indices(l, ii):
    ii = set(ii)
    return [e for i, e in enumerate(l) if i not in ii]


# Axes

class Axis(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        #self.size = size

indep_observation_axis_type, joint_axis_type, sample_space_axis_type, MC_axis_type = range(4)

observations_axis = Axis("observations", indep_observation_axis_type)
categories_axis = Axis("categories", joint_axis_type)
segments_axis = Axis("segments", sample_space_axis_type)
didBuy_axis = Axis("didBuy", sample_space_axis_type)



'''
# Array Container

Why use this?
1. Storing custom "axis" information with multi-dimensional array
2. Imitating mutable changes (e.g., removing a row in-place)
'''
class Arr(object):
    def __init__(self, arr, axes):
        self.arr = arr
        self.axes = axes

    def rearranged(self, axes, negate=False):
        is_in = [self.axes.index(ax) for ax in axes]
        is_not_in = [i for i, ax in enumerate(self.axes) if ax not in axes]
        include_ind = is_not_in if negate else is_in
        exclude_ind = is_in if negate else is_not_in
        new_order = include_ind + exclude_ind

        result = np.transpose(self.arr, new_order)
        m = len(include_ind)
        new_part = zip(range(m), [self.axes[i] for i in include_ind])
        old_part = zip(range(m, m + len(exclude_ind)), [self.axes[i] for i in exclude_ind])
        return result, new_part, old_part

    def remove(self, indices, from_axis):
        if from_axis not in self.axes:
            raise Exception()
        if len(indices) == 0:
            return np.array([])
        axis_ind = self.axes.index(from_axis)
        removed = np.take(self.arr, indices, axis=axis_ind)
        self.arr = np.delete(self.arr, indices, axis_ind)
        return removed

# Count Data (convenience object)

class CountData(Arr):
    def __init__(self, data_frame):
        counts = data_frame.values
        N_all, K = counts.shape
        self.N_all = N_all
        self.N = N_all
        self.num_categories = K
        self.freqs = np.ones(self.N)
        self.counts = counts # todo: utilize freqs

        self.axes = [observations_axis, categories_axis]
        self.arr = counts

        self.excluded_observations = list()

    def exclude_observations(self, indices=[]):
        self.excluded_observations.append(self.remove(indices=indices, from_axis=observations_axis))



class Conditionals(Arr):

    def __init__(self, arr, axes, on_bad_observations_trigger):
        self.arr = arr
        self.axes = axes
        self.on_bad_observations_trigger = on_bad_observations_trigger

        self._normalized_conditionals = {}

    def get_normalized_conditionals(self, axes_to_sum_over):
        cache_key = "by_" + "_".join([ax.name for ax in axes_to_sum_over])
        if cache_key not in self._normalized_conditionals:
            arr, keep_part, sum_over_part  = self.rearranged(axes=axes_to_sum_over, negate=True)
            sum_over_inds = [i for i,_ in sum_over_part]
            sums = arr.sum(axis=tuple(sum_over_inds))
            out = arr / sums.reshape(list(sums.shape) + [1 for _ in sum_over_inds])

            isnan = np.isnan(out) # find and remove problematic observations
            if isnan.any():
                observations_axis_ind = [i for i, ax in keep_part + sum_over_part if ax == observations_axis][0]
                bad_obs_inds = list(set(np.where(isnan)[observations_axis_ind]))
                out = np.delete(out, list(bad_obs_inds), observations_axis_ind)
                self.remove(bad_obs_inds, from_axis=observations_axis)
                self.on_bad_observations_trigger(bad_obs_inds)

            self._normalized_conditionals[cache_key] = Conditionals(out, self.axes, self.on_bad_observations_trigger)
        return self._normalized_conditionals[cache_key]

    def get_arr(self, axes, apply_func=np.mean):
        out, _, exclude = self.rearranged(axes=axes)
        if len(exclude) > 0:
            out = np.apply_over_axes(apply_func, out, [i for i, _ in exclude])
        return out

    def get_sums(self, axis): # todo: remove because redundant given get_arr
        exclude_ind = [i for i, ax in enumerate(self.axes) if ax != axis]
        return self.table.sum(axis=tuple(exclude_ind))


#
# class BuyComponent(NetworkComponent):
#     prob_table_axes = [segments_axis, categories_axis, didBuy_axis]
#     prob_table_dim_names = ["segments", "categories", "didBuy"]
#
#     def __init__(self, num_segments, count_data):
#         N, K = count_data.shape # count_data must be N-by-K, where K = number of categories
#         self.num_segments = num_segments
#         self.num_categories = K
#
#         # params is C-by-K matrix of "buy probabilities" of making a purchase in category K being in segment C
#         self.params = ((count_data>0).sum(axis=0)/N).repeat(num_segments).reshape([num_segments, K], order='F')
#
#     def prob_table(self, data):
#         return np.transpose(np.array([self.params, 1-self.params]), [1, 2, 0])
#
#
#
# class InflatedZerosPoissonComponent(PoissonComponent):
#     prob_table_axes = [segments_axis, didBuy_axis, observations_axis, categories_axis]
#     prob_table_dim_names = ["segments", "didBuy", "observations", "categories"]
#
#     def initial_estimates(self, count_data):
#         return count_data.sum(axis=0) / (count_data>0).sum(axis=0)
#
#
#
#     def prob_table(self, count_data):
#         # count_data is N-by-K
#         # must return S-by-2-by-N-by-K
#         poisson = self.poisson_prob_table(count_data, self.params) # S-by-N-by-K
#
#         counts = count_data.reshape([1] + list(count_data.shape))
#         no_buy = np.power(poisson * 0, counts) # S-by-N-by-K
#
#         return np.array([poisson, no_buy]).swapaxes(0,1)