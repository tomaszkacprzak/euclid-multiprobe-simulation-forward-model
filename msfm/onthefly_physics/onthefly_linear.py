# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""

from msfm.utils import logger
import warnings
from msfm.onthefly_pipeline import OntheflyPipeline
from msfm.utils import parameters, prior

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

class OntheflyPhysicsModelLinear(OntheflyPipeline):

    def __init__(self, conf):
        self.conf = conf

        astro_params = self.conf["analysis"]["params"]["ia"]["nla"]
        astro_params += self.conf["analysis"]["params"]["bg"]["linear"]
        astro_params += self.conf["analysis"]["params"]["sc"]
        LOGGER.info(f"Astrophysical parameters: {astro_params}")
        self.astro_priors = parameters.get_prior_intervals(astro_params, conf=self.conf)

    def sample_astro_parameters(self, i_cosmo, num_examples):
        return prior.sample_astro_parameters_latin_hypercube(self.astro_params, i_cosmo, num_examples, self.astro_priors)

    def get_dataset(self):
        dataset = super().get_dataset(self.conf)

    def augmentations(self):

        pass



    def get_loader(self):
        
        raise NotImplementedError("Not implemented")



astro_samples = prior.sample_astro_parameters(astro_params, i_cosmo, n_examples_per_cosmo, n_noise_per_signal, astro_priors)
