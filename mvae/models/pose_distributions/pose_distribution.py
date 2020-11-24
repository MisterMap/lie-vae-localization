class PoseDistribution(object):
    @property
    def mean_dimension(self):
        raise NotImplementedError

    @property
    def logvar_dimension(self):
        raise NotImplementedError

    def sample(self, mean, logvar):
        raise NotImplementedError

    def log_prob(self, value, mean, logvar):
        raise NotImplementedError
