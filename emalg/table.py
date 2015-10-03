from __future__ import division
import numpy as np
import logging
import sys

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
        return out_table_class(self._scaled_cache[cache_key], self.axes)

    def _on_scaled_nan(self, scaled_arr):
        raise Exception() # Absract method (Base class Table does not know what to do with scaled NaNs

    def remove(self, indices, from_axis):
        if from_axis not in self.axes:
            raise Exception()
        if len(indices) == 0:
            return np.array([])
        axis_ind = self.axes.index(from_axis)
        removed = np.take(self.arr, indices, axis=axis_ind)
        self.arr = np.delete(self.arr, indices, axis_ind)
        return removed

class Conditionals(Table):

    def __init__(self, arr, axes):
        super(Conditionals, self).__init__(arr, axes)
        self.on_bad_observations_trigger = lambda *args: None

    def _on_scaled_nan(self, scaled_arr):
        try:
            observations_axis_ind = [i for i, ax in enumerate(self.axes) if ax == observations_axis][0]
        except IndexError as e:
            logging.error("Caught NaNs in a table of conditionals with Observations axis missing.")
            sys.exit(1)

        isnan = np.isnan(scaled_arr)
        bad_obs_inds = list(set(np.where(isnan)[observations_axis_ind]))
        out = np.delete(scaled_arr, list(bad_obs_inds), observations_axis_ind)
        self.remove(bad_obs_inds, from_axis=observations_axis)
        self.on_bad_observations_trigger(bad_obs_inds)
        return out

# Count Data (convenience object)

class CountData(Table):
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

