from __future__ import division
from table import *
from scipy.stats import multivariate_normal, chi2
from numpy import random
from model_component import ModelComponent
from segment import SegmentComponent
from matplotlib.patches import Ellipse
import pandas as pd
import matplotlib.pyplot as plt

elements_axis = Axis("elements", joint_indep_axis_type)

# PLOTTING OF ELLIPSES
# borrowed from http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/
def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[1, 0, 0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    kwrg = {'facecolor': fc, 'edgecolor': ec, 'alpha': a, 'linewidth': lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)


"""
Gaussian Mixture
"""
class GaussianMixtureComponent(ModelComponent):
    conditionals_axes = [observations_axis, segments_axis]

    def __init__(self, data, num_segments=2):
        self.data = data
        super(GaussianMixtureComponent, self).__init__(data, num_segments)

        self.mus, self.sigmas = self.generate_initial_estimates()

    def generate_initial_estimates(self):
        N, K = self.data.arr.shape

        # mus is S-by-K matrix
        mus = self.data.arr.mean(axis=0)
        mus = mus.repeat(self.num_segments).reshape([self.num_segments, K], order='F')
        mus += random.ranf(size=mus.size).reshape(mus.shape)

        # sigmas is S-by-K-by-K array
        sigmas = np.eye(K).repeat(self.num_segments).reshape([self.num_segments, K, K], order='F')
        return mus, sigmas

    def params(self):
        return self.mus, self.sigmas

    def get_conditionals(self):
        # data must be N-by-K matrix
        # returns N-by-S probability table of observed vectors (given each segment)
        N, K = self.data.arr.shape
        tmp_list = list()
        for s in range(self.num_segments):
            mus = self.mus[s,:]
            cov = self.sigmas[s,:,:]
            pdf = multivariate_normal(mus, cov).pdf(self.data.arr) # array of length N
            tmp_list.append(pdf.reshape([N, 1]))
        return Table(np.hstack(tmp_list), axes=self.conditionals_axes)

    def use_posteriors(self, posteriors, obtained_from):
        super(GaussianMixtureComponent, self).use_posteriors(posteriors, obtained_from)

        if isinstance(obtained_from, SegmentComponent):
            N, K = self.data.arr.shape
            c_mat = posteriors.get_arr([segments_axis, observations_axis])  # S-by-N
            c_sums = posteriors.get_arr([segments_axis], apply_func=np.sum)  # S-vec
            SC = np.dot(c_mat, self.data.arr)  # SC is S-by-K
            S = len(c_sums)
            self.mus = SC / c_sums.reshape(S, 1) # S-by-K
            if np.isnan(self.mus).any():
                pass  # debug purposes

            # the following needs to be optimized (loops to be vectorized)
            cov_matrices = list()
            for s in range(self.num_segments):
                mu_s = self.mus[s, :].reshape(1, K) # 1-by-K
                diff = self.data.arr - mu_s
                cov_s = np.zeros(K * K).reshape(K, K)
                for n in range(N):
                    diff_n = diff[n,:].reshape(K, 1)
                    cov_s += np.dot(diff_n, diff_n.transpose()) * c_mat[s, n] / c_sums[s]
                cov_matrices.append(cov_s)
            self.sigmas = np.array(cov_matrices)

    def use_params_for_fitted_info(self, params, from_component):
        if isinstance(from_component, SegmentComponent):
            self.segment_names = from_component.segment_names()

    def fitted_info(self):
        _, K = self.data.arr.shape
        cols = ["Mean X_%d" % d for d in range(1, K + 1)]
        means = self.mus
        return pd.DataFrame(np.round(means, decimals=4), index=self.segment_names, columns=cols)

    def use_params_for_fitted_plots(self, params, from_component):
        if isinstance(from_component, SegmentComponent):
            self.segment_probs = from_component.params()

    def fitted_plots(self):
        N, K = self.data.arr.shape
        if K == 2:
            fig = plt.figure()
            xy = self.data.arr
            plt.plot(xy[:, 0], xy[:, 1], 'o', zorder=4, color='darkgray')
            plt.gca().set_axis_bgcolor((0.98, 0.98, 0.99))

            for s in range(self.num_segments):
                mus = self.mus[s, :]
                cov = self.sigmas[s, :, :]
                plot_cov_ellipse(cov, mus, volume=0.8)

            return [fig]

        return []
