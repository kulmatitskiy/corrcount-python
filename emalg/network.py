from components import *
from base import *

class Network(object):
    def __init__(self, component_classes, count_data, num_segments):
        self.num_segments = num_segments
        self.count_data = count_data
        self.components = [C(count_data=count_data, num_segments=num_segments) for C in component_classes]

    def run_once(self):
        def on_bad_observations(indices):
                self.count_data.remove(indices, from_axis=observations_axis)

        reversed_components = list(reversed(self.components))
        for i, component in enumerate(reversed_components):
            if i == 0:
                prob_table = component.prob_table()
                axes = component.prob_table_axes
                # tmp = component.get_prob_arr() # Arr(prob_table, axes)
            else:
                axes_keep = component.prob_table_axes

                axes_to_sum_over = [i for i, ax in enumerate(axes) if ax not in axes_keep and ax.type == sample_space_axis_type]
                if len(axes_to_sum_over) > 0:
                    prob_table = prob_table.sum(axis=tuple(axes_to_sum_over))
                    axes = remove_indices(axes, axes_to_sum_over)

                axes_to_mult_over = [i for i, ax in enumerate(axes) if ax not in axes_keep and ax.type == joint_axis_type]
                if len(axes_to_mult_over) > 0:
                    prob_table = prob_table.prod(axis=tuple(axes_to_mult_over))
                    axes = remove_indices(axes, axes_to_mult_over)

                # todo: axes_to_average_over (for MC trials)

                prob_table = np.transpose(prob_table, tuple([axes.index(ax) for ax in axes_keep]))
                prob_table = prob_table * component.prob_table()
                axes = axes_keep

            conditionals = Conditionals(prob_table, axes, on_bad_observations_trigger=on_bad_observations)
            for subseq_component in reversed_components[:(i+1)]:
                subseq_component.use(conditionals, component)

        print('\n-----------\n')
