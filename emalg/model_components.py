from numpy import random
from table import *
from scipy.stats import poisson


class ModelComponent(object):
    main_axis = None
    is_latent = False
    def __init__(self, count_data, num_segments=2):
        super(ModelComponent, self).__init__()
        self.count_data = count_data
        self.num_segments = num_segments

    def params(self):
        return NotImplementedError()

    def get_conditionals(self):
        return NotImplementedError()

    def use_posteriors(self, posteriors, obtained_from):
        pass


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
        if obtained_from is self:
            n, _ = self.count_data.arr.shape
            self.seg_probs = posteriors.get_arr([segments_axis], apply_func=np.sum) / n


class PoissonComponentWithLatentClasses(ModelComponent):
    conditionals_axes = [segments_axis, observations_axis, categories_axis]

    def __init__(self, count_data, num_segments=2):
        super(PoissonComponentWithLatentClasses, self).__init__(count_data, num_segments)
        self.lambdas = self.generate_initial_estimates()

    def generate_initial_estimates(self):
        # lambdas is S-by-K matrix
        K = self.count_data.num_categories
        lambdas = self.global_means_by_category().repeat(self.num_segments).reshape([self.num_segments, K], order='F')
        lambdas += random.ranf(size=lambdas.size).reshape(lambdas.shape)
        return lambdas

    def global_means_by_category(self):
        return self.count_data.arr.mean(axis=0) # todo: use freqs

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
        return Table(self.poisson_prob_arr(self.lambdas), axes=self.conditionals_axes)

    def use_posteriors(self, posteriors, obtained_from):
        if isinstance(obtained_from, SegmentComponent):
            segm_posteriors_mat = posteriors.get_arr([segments_axis, observations_axis])  # S-by-N
            segm_posteriors_sums = posteriors.get_arr([segments_axis], apply_func=np.sum) # S-vec

            S = len(segm_posteriors_sums)

            SC = np.dot(segm_posteriors_mat, self.count_data.arr) # SC is S-by-K
            self.lambdas = SC / segm_posteriors_sums.reshape(S, 1)





#
# class BuyComponent(NetworkComponent):
#     conditionals_axes = [segments_axis, categories_axis, didBuy_axis]
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
#     def get_conditionals(self, data):
#         return np.transpose(np.array([self.params, 1-self.params]), [1, 2, 0])
#
#
#
# class InflatedZerosPoissonComponent(PoissonComponent):
#     conditionals_axes = [segments_axis, didBuy_axis, observations_axis, categories_axis]
#     prob_table_dim_names = ["segments", "didBuy", "observations", "categories"]
#
#     def initial_estimates(self, count_data):
#         return count_data.sum(axis=0) / (count_data>0).sum(axis=0)
#
#
#
#     def get_conditionals(self, count_data):
#         # count_data is N-by-K
#         # must return S-by-2-by-N-by-K
#         poisson = self.poisson_prob_table(count_data, self.params) # S-by-N-by-K
#
#         counts = count_data.reshape([1] + list(count_data.shape))
#         no_buy = np.power(poisson * 0, counts) # S-by-N-by-K
#
#         return np.array([poisson, no_buy]).swapaxes(0,1)


