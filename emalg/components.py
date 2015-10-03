from misc import *
from table import *


class Component(object):
    prob_table_axes = list()

    def __init__(self):
        pass

    def process_input(self, table, for_component):
        raise NotImplementedError()


class ExpectedLogLikelihood(Component):
    def __init__(self, count_data, store_trace=False):
        super(ExpectedLogLikelihood, self).__init__()
        self.count_data = count_data
        self.ll = 0
        self.trace = None

    def process_input(self, input_table, for_component):
        if for_component is self:
            if self.trace is None:
                self.trace = list()
            else:
                self.trace.append(self.ll)
            self.ll = 0
        elif input_table is not None:
            n = self.count_data.N_all
            self.ll += input_table.arr.sum() / n # todo: utilize freqs

        return input_table


class ConditionalsMaker(Component):
    def __init__(self, on_bad_observations_trigger):
        super(ConditionalsMaker, self).__init__()
        self.on_bad_observations_trigger = on_bad_observations_trigger

    def process_input(self, input_table, for_component):
        if for_component is self:
            return input_table

        if input_table is None:
            return for_component.prob_table()

        axes = input_table.axes
        tmp_prob_arr = input_table.arr
        next_prob_table = for_component.prob_table()
        axes_keep = next_prob_table.axes

        axes_to_sum_over = [i for i, ax in enumerate(axes) if ax not in axes_keep and ax.type == sample_space_axis_type]
        if len(axes_to_sum_over) > 0:
            tmp_prob_arr = tmp_prob_arr.sum(axis=tuple(axes_to_sum_over))
            axes = remove_indices(axes, axes_to_sum_over)

        axes_to_mult_over = [i for i, ax in enumerate(axes) if ax not in axes_keep and ax.type == joint_axis_type]
        if len(axes_to_mult_over) > 0:
            tmp_prob_arr = tmp_prob_arr.prod(axis=tuple(axes_to_mult_over))
            axes = remove_indices(axes, axes_to_mult_over)

        # todo: axes_to_average_over (for MC trials)

        tmp_prob_arr = np.transpose(tmp_prob_arr, tuple([axes.index(ax) for ax in axes_keep]))
        out_prob_arr = tmp_prob_arr * next_prob_table.arr
        out_axes = axes_keep

        out_conditionals = Conditionals(out_prob_arr, out_axes)
        out_conditionals.on_bad_observations_trigger = self.on_bad_observations_trigger
        return out_conditionals


