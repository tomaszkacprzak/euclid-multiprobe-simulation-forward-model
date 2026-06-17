# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""
import warnings
import numpy as np
import healpy as hp
import torch
from torch.utils.data import DataLoader

from msfm.utils import logger
from msfm.onthefly_pipeline import OntheflyPipeline
from msfm.utils import parameters, prior, clustering

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

class OntheflyPhysicsModelLinear(OntheflyPipeline):

    def __init__(self, conf, num_samples_prior=1_000_000):
        
        self.conf = conf
        self.astro_samples = self.get_astro_params(num_samples_prior)
        self.shape_noise = 0.3
        self.num_gal_wl = torch.from_numpy(np.array(self.conf["survey"]["WL"]["n_gal"]))
        self.num_gal_gc = torch.from_numpy(np.array(self.conf["survey"]["GC"]["n_gal"]))
        self.pixel_area = hp.nside2pixarea(self.conf["analysis"]["n_side"], degrees=True)
        
    def get_dataset(self, webds_pattern):

        dataset = super().get_dataset(webds_pattern)
        dataset = dataset.map(self.physics_transform)

        return dataset

    def get_astro_params(self, num_samples_prior):
        
        # Read pre-computed parameters from configs
        self.cosmo_params = self.conf["analysis"]["params"]["cosmo"]
        self.bary_params = self.conf["analysis"]["params"]["bary"]

        # Create on-the-fly parameters from config
        self.astro_params  = self.conf["analysis"]["params"]["ia"]["nla"]
        self.astro_params += self.conf["analysis"]["params"]["bg"]["linear"]
        self.astro_params += self.conf["analysis"]["params"]["sc"]

        # Indices of parameters
        self.inds_cosmo_params = {key:i for i, key in enumerate(self.cosmo_params)}
        self.inds_bary_params = {key:i for i, key in enumerate(self.bary_params)}
        self.inds_astro_params = {key:i for i, key in enumerate(self.astro_params)}
        self.inds_all_cosmo_params = {key:i for i, key in enumerate(self.cosmo_params)}
        self.inds_all_bary_params = {key:i+len(self.cosmo_params) for i, key in enumerate(self.bary_params)}
        self.inds_all_astro_params = {key:i+len(self.cosmo_params)+len(self.bary_params) for i, key in enumerate(self.astro_params)}
        self.inds_all_params = {**self.inds_all_cosmo_params, **self.inds_all_bary_params, **self.inds_all_astro_params}
        self.all_params = self.cosmo_params + self.bary_params + self.astro_params
        LOGGER.info(f"All parameters: {self.all_params}")
        LOGGER.info(f"Indices of all parameters: {self.inds_all_params}")

        # Pre-generate samples from latin hypercube
        self.astro_priors = parameters.get_prior_intervals(self.astro_params, conf=self.conf)
        self.astro_samples = prior.sample_astro_parameters_latin_hypercube(
            self.astro_params, 
            seed=self.conf['master_seed']+424344, 
            n_examples=num_samples_prior, 
            astro_priors=self.astro_priors)

        LOGGER.info(f"Sampled astrophysical parameters: {self.astro_samples.shape}")
        for i, param_name in enumerate(self.astro_params):
            p_ = self.astro_samples[:, i]
            LOGGER.info(f"   {param_name:>20s}  min={np.min(p_): 8.3f} max={np.max(p_): 8.3f} mean={np.mean(p_): 8.3f}")

        return torch.from_numpy(self.astro_samples)

    def sample_astro_parameters(self):

        j = torch.randint(0, self.astro_samples.shape[0], (1,))
        return self.astro_samples[j].squeeze()

    def physics_transform(self, example):

        # unpack the example
        gg, ga, gd, ds, dg, qg, cosmo, i_sobol, i_signal, n_params, n_pix, n_z_wl, n_z_gc = example
        astro_params = self.sample_astro_parameters()
        targets = torch.cat([cosmo, astro_params], dim=0)

        #
        # Linear bias map for galaxy clustering
        #
        ng_bar = self.num_gal_wl * self.pixel_area
        ids_bg = [self.inds_astro_params[key] for key in self.conf["analysis"]["params"]["bg"]["linear"]]
        bg = astro_params[ids_bg]

        assert dg.shape[-1] == len(bg), "The number of bias parameters must match the number of tomographic bins"
        ng_lambda = clustering.galaxy_density_to_count(ng_bar, dg, bg=bg, qdg=None, qbg=None, mg=None, cg=None, systematics_map=None, mask=None, backend='torch')
        ng = torch.poisson(ng_lambda)

        LOGGER.info(f'drawn poisson galaxy clustering map ng={ng.shape} min={ng.min():>10.3f} max={ng.max():>10.3f} mean={ng.mean():>10.3f}')

        return ng, targets

      


# astro_samples = prior.sample_astro_parameters(astro_params, i_cosmo, n_examples_per_cosmo, n_noise_per_signal, astro_priors)
