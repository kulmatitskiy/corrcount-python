from components import *

class Network(object):
    def __init__(self, component_classes, count_data, num_segments):
        self.num_segments = num_segments
        self.count_data = count_data
        self.components = [C(count_data=count_data, num_segments=num_segments) for C in component_classes]

        def on_bad_observations(indices):
            self.count_data.remove(indices, from_axis=observations_axis)

        self.components.append(ConditionalsMaker(on_bad_observations_trigger=on_bad_observations))

        self.expected_ll_component = ExpectedLogLikelihood(count_data, store_trace=True)
        self.components.append(self.expected_ll_component)

    def run_once(self):
        reversed_components = list(reversed(self.components))
        table = None
        for i, component in enumerate(reversed_components):
            for subseq_component in reversed_components[:(i+1)]:
                table = subseq_component.process_input(table, for_component=component)
        #print(self.expected_ll_component.ll)
        #print('\n-----------\n')
