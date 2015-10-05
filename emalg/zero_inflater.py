from table import *
import pandas as pd
from model_component import ModelComponent
from segment import SegmentComponent
from numpy import random

class ZeroInflaterComponent(ModelComponent):
    main_axis = didBuy_axis
    is_latent = True
    conditionals_axes = [segments_axis, didBuy_axis, observations_axis, categories_axis]

    def __init__(self, count_data, num_segments=2):
        super(ZeroInflaterComponent, self).__init__(count_data, num_segments)
        self.deltas = self.generate_initial_estimates()

    def generate_initial_estimates(self):
        # deltas is S-by-K matrix
        k = self.count_data.num_categories
        non_zero_means = (self.count_data.arr > 0).sum(axis=0).astype('float64') / self.count_data.N_all
        deltas = non_zero_means.repeat(self.num_segments).reshape([self.num_segments, k], order='F')
        deltas += (random.ranf(size=deltas.size)*0.05).reshape(deltas.shape)
        deltas[deltas >= 1] = 0.95
        return deltas

    def params(self):
        return self.deltas

    def get_conditionals(self):
        # self.deltas is S-by-K matrix, we need to output S-by-2-by-N-by-K array
        n, k = self.count_data.arr.shape
        tmp = np.array([self.deltas, 1 - self.deltas]) # 2-by-S-by-K
        tmp = tmp.repeat(n).reshape([n, 2, self.num_segments, k], order='F')
        tmp = np.transpose(tmp, axes=[2, 1, 0, 3])
        return Table(tmp, axes=self.conditionals_axes)

    def use_posteriors(self, posteriors, obtained_from):
        super(ZeroInflaterComponent, self).use_posteriors(posteriors, obtained_from)

        if obtained_from is self:
            self.deflate_posteriors_arr = self.view_of_deflate_posteriors(posteriors)

        elif isinstance(obtained_from, SegmentComponent):
            seg_posteriors_arr = posteriors.get_arr([segments_axis, observations_axis])  # S-by-N
            tmp = np.einsum('ijk,ij->ik', self.deflate_posteriors_arr, seg_posteriors_arr)  # S-by-K
            tmp = tmp / seg_posteriors_arr.sum(axis=1).reshape([self.num_segments, 1])  # S-by-K
            tmp[tmp >= 1] = 1 - 1e-8  # zero or one probabilities are not allowed
            self.deltas = tmp

    def view_of_deflate_posteriors(self, all_inflater_posteriors):
        tmp = np.take(all_inflater_posteriors.arr, [0], axis=1) # just take the deflation part
        tmp = tmp.reshape([self.num_segments] + list(self.count_data.arr.shape), order='F')
        return tmp  # S-by-N-by-K

    def use_params_for_fitted_info(self, params, from_component):
        if isinstance(from_component, SegmentComponent):
            self.segment_names = from_component.segment_names()

    def fitted_info(self):
        return pd.DataFrame(data=np.round(self.deltas, 2), index=self.segment_names, columns=self.count_data.category_names)

