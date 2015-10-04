from table import *
from model_component import ModelComponent
import numpy as np
import pandas as pd
from numpy import random

class SegmentComponent(ModelComponent):
    main_axis = segments_axis
    is_latent = True
    conditionals_axes = [observations_axis, segments_axis]

    def __init__(self, count_data, num_segments=2):
        super(SegmentComponent, self).__init__(count_data, num_segments)
        self.seg_probs = self.generate_initial_estimates()

    def generate_initial_estimates(self):
        # TODO: init params using data (k-means?)
        tmp = 0.8 * random.ranf(size=self.num_segments) + 0.1
        return tmp / tmp.sum()

    def params(self):
        return self.seg_probs

    def get_conditionals(self):
        n, _ = self.count_data.arr.shape
        return Table(self.seg_probs.repeat(n).reshape(n, self.num_segments, order='F'), axes=self.conditionals_axes)

    def use_posteriors(self, posteriors, obtained_from):
        super(SegmentComponent, self).use_posteriors(posteriors, obtained_from)
        if obtained_from is self:
            n, _ = self.count_data.arr.shape
            self.seg_probs = posteriors.get_arr([segments_axis], apply_func=np.sum) / n

    def segment_names(self):
        return ["S%02d" % s for s in range(1,self.num_segments+1)]

    def fitted_info(self):
        segms = self.segment_names()
        col1 = np.round(self.seg_probs, decimals=2)
        col2 = np.round(self.seg_probs * self.count_data.N_all)
        return pd.DataFrame(data=np.hstack((col1, col2)), index=segms, columns=["Probability", "Size in sample"])


