
class ModelComponent(object):
    main_axis = None
    is_latent = False

    def __init__(self, count_data, num_segments=2):
        super(ModelComponent, self).__init__()
        self.count_data = count_data
        self.num_segments = num_segments
        self.saved_params = None

    def params(self):
        raise NotImplementedError()

    def save_params(self):
        self.saved_params = self.params()

    def get_conditionals(self):
        raise NotImplementedError()

    def use_posteriors(self, posteriors, obtained_from):
        if obtained_from is self:
            self.posteriors = posteriors

    def use_params_for_fitted_info(self, params, from_component):
        pass

    def fitted_info(self):
        return None

    def use_params_for_fitted_plots(self, params, from_component):
        pass

    def fitted_plots(self):
        pass





