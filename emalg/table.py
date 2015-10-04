from __future__ import division
import numpy as np
from misc import *

# Axes

class Axis(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type

indep_observation_axis_type, joint_axis_type, sample_space_axis_type, MC_axis_type = range(4)

observations_axis = Axis("observations", indep_observation_axis_type)
categories_axis = Axis("categories", joint_axis_type)
segments_axis = Axis("segments", sample_space_axis_type)
didBuy_axis = Axis("didBuy", sample_space_axis_type)


class MissingObservationAxisError(Exception):
    pass


'''
# Table = Array Container with Custom Axes

Why use this?
1. Storing custom "axis" information with multi-dimensional array
2. Imitating mutable changes (e.g., removing a row in-place)
'''
class Table(object):
    def __init__(self, arr, axes):
        self.arr = arr
        self.axes = axes

        self._scaled_cache = {}

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

    def get_arr(self, axes, apply_func=np.mean):
        out, _, exclude = self.rearranged(axes=axes)
        if len(exclude) > 0:
            out = np.apply_over_axes(apply_func, out, [i for i, _ in exclude])
        return out

    def get_scaled(self, axes_to_sum_over, out_table_class=None):
        if out_table_class is None:
            out_table_class = self.__class__

        cache_key = "by_" + "_".join([ax.name for ax in axes_to_sum_over])

        if cache_key not in self._scaled_cache:
            arr, keep_part, sum_over_part  = self.rearranged(axes=axes_to_sum_over, negate=True)
            sum_over_inds = [i for i,_ in sum_over_part]
            sums = arr.sum(axis=tuple(sum_over_inds))
            out = arr / sums.reshape(list(sums.shape) + [1 for _ in sum_over_inds])

            map_to_original_axes_order = [(i, self.axes.index(ax)) for i, ax in keep_part+sum_over_part]
            out = np.transpose(out, tuple([i for i, i_orig in sorted(map_to_original_axes_order, key=lambda tpl:tpl[1])]))

            if np.isnan(out).any():
                out = self._on_scaled_nan(out)

            self._scaled_cache[cache_key] = out
            #Conditionals(out, self.axes, self.on_bad_observations_trigger)
        return Table(self._scaled_cache[cache_key], self.axes)

    def _on_scaled_nan(self, scaled_arr):
        raise Exception() # Absract method (Base class Table does not know what to do with scaled NaNs

    def get_observations_inds(self, boolean_mask):
        if observations_axis not in self.axes:
            raise MissingObservationAxisError()

        observations_axis_ind = self.axes.index(observations_axis)
        required_obs_inds = list(set(np.where(boolean_mask)[observations_axis_ind]))
        return observations_axis_ind, required_obs_inds

    def remove(self, indices, from_axis):
        if from_axis not in self.axes:
            raise Exception()
        if len(indices) == 0:
            return np.array([])
        axis_ind = self.axes.index(from_axis)
        removed = np.take(self.arr, indices, axis=axis_ind)
        self.arr = np.delete(self.arr, indices, axis_ind)
        return removed


class CountData(Table):
    def __init__(self, data_frame):
        counts = data_frame.values
        N_all, K = counts.shape
        self.category_names = data_frame.columns
        self.N_all = N_all
        self.N = N_all
        self.num_categories = K
        self.freqs = np.ones(self.N)
        self.counts = counts # todo: utilize freqs

        self.axes = [observations_axis, categories_axis]
        self.arr = counts

        self.excluded_observations = list()

    def number_of_observations(self):
        n, k = self.arr.shape
        return n # todo: use freqs

    def exclude_observations(self, indices=[]):
        self.excluded_observations.append(self.remove(indices=indices, from_axis=observations_axis))

    def remove(self, indices, from_axis):
        if from_axis is not observations_axis:
            raise Exception
        else:
            removed = super(CountData, self).remove(indices, from_axis=observations_axis)
            self.excluded_observations.append(removed)


class ProbTable(Table):
    def __init__(self, count_data, conditionals):
        self.count_data = count_data
        super(ProbTable, self).__init__(conditionals.arr, conditionals.axes)

    def update(self, conditionals):
        axes = self.axes
        axes_to_keep = conditionals.axes
        tmp_prob_arr = self.arr

        axes_to_sum_over = [i for i, ax in enumerate(axes) if ax not in axes_to_keep and ax.type == sample_space_axis_type]
        if len(axes_to_sum_over) > 0:
            tmp_prob_arr = tmp_prob_arr.sum(axis=tuple(axes_to_sum_over))
            axes = remove_indices(axes, axes_to_sum_over)

        axes_to_mult_over = [i for i, ax in enumerate(axes) if ax not in axes_to_keep and ax.type == joint_axis_type]
        if len(axes_to_mult_over) > 0:
            tmp_prob_arr = tmp_prob_arr.prod(axis=tuple(axes_to_mult_over))
            axes = remove_indices(axes, axes_to_mult_over)

        tmp_prob_arr = np.transpose(tmp_prob_arr, tuple([axes.index(ax) for ax in axes_to_keep]))
        self.arr = tmp_prob_arr * conditionals.arr
        self.axes = axes_to_keep


class ExpLLTable(Table):
    def __init__(self, count_data, conditionals):
        self.count_data = count_data
        super(ExpLLTable, self).__init__(np.log(conditionals.arr), conditionals.axes)

    def update(self, conditionals, posteriors):
        axes = self.axes
        tmp_expll_arr = self.arr
        axes_to_keep = conditionals.axes

        axes_to_sum_over = [i for i, ax in enumerate(axes) if ax not in axes_to_keep and ax.type in [sample_space_axis_type, joint_axis_type]]
        if len(axes_to_sum_over) > 0:
            tmp_expll_arr = tmp_expll_arr.sum(axis=tuple(axes_to_sum_over))
            axes = remove_indices(axes, axes_to_sum_over)

        # todo: axes_to_average_over (for MC trials)

        tmp_expll_arr = np.transpose(tmp_expll_arr, tuple([axes.index(ax) for ax in axes_to_keep]))
        tmp_expll_arr += np.log(conditionals.arr)
        self.arr = tmp_expll_arr * posteriors.arr
        self.axes = axes_to_keep

    def get_ll(self):
        return self.arr.sum() / self.count_data.N_all
