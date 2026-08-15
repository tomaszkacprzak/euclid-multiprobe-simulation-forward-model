# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""
# from tkinter import E
import warnings
import numpy as np
import healpy as hp
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader


from msfm.utils import logger
from msfm.onthefly_pipeline import OntheflyPipeline
# from msfm.onthefly_physics.onthefly_base import OntheflyPhysicsModel
from msfm.utils import parameters, prior, clustering, redshift, files

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

class OntheflyPhysicsModelLinear(nn.Module):

    def __init__(self, conf, scalers=False, seed=424344, num_samples_prior=1_000_000, device=None, nside=None, **kwargs):

        super().__init__()

        self.conf = conf
        self.seed = seed
        self.device = device
        self.nside = nside
        self.astro_samples = self.get_astro_params(num_samples_prior)
        self.shape_noise_std = 0.03
        self.num_gal_wl = torch.from_numpy(np.array(self.conf["survey"]["WL"]["n_gal"])).to(self.device)
        self.num_gal_gc = torch.from_numpy(np.array(self.conf["survey"]["GC"]["n_gal"])).to(self.device)
        self.pixel_area = hp.nside2pixarea(self.nside, degrees=True)
        self.sample_uniform_lo = torch.tensor(0., device=self.device, dtype=torch.float32)
        self.sample_uniform_hi = torch.tensor(2 * math.pi, device=self.device, dtype=torch.float32)
        self.num_targets = len(self.all_params)
        self.num_channels = 24
        # self.param_names = ['Om', 's8', 'Ob', 'H0', 'ns', 'w0', 'bary_Mc', 'bary_nu', 'Aia', 'n_Aia', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']
        self.scalers = scalers
        if self.scalers:
            self.set_scalers()
        LOGGER.info(f"Created physics model Linear, num_channels={self.num_channels}, num_targets={self.num_targets}, apply_scalers={self.scalers}, shape_noise_std={self.shape_noise_std}")
        

    def set_scalers(self):

        # ['Om', 's8', 'Ob', 'H0', 'ns', 'w0', 'bary_Mc', 'bary_nu', 'Aia', 'n_Aia', 'bg1', 'bg2', 'bg3', 'bg4', 'bg5', 'bg6', 'bsc1', 'bsc2', 'bsc3', 'bsc4', 'bsc5', 'bsc6']

        shift_channels = 0.
        scale_channels = torch.tensor([1.]*12 + [1./100000.]*6 + [1./10000.]*6, device=self.device)

        list_min = torch.tensor([self.conf['analysis']['grid']['priors'][p][0] for p in self.all_params], device=self.device, dtype=torch.float32)
        list_max = torch.tensor([self.conf['analysis']['grid']['priors'][p][1] for p in self.all_params], device=self.device, dtype=torch.float32)
        shift_targets = -list_min
        scale_targets = 1./(list_max-list_min)

        self.targets_shift = shift_targets.reshape(1, -1).to(self.device)
        self.targets_scale = scale_targets.reshape(1, -1).to(self.device)
        self.channel_shift = shift_channels #.reshape(1, 1, -1).to(self.device)
        self.channel_scale = scale_channels.reshape(1, 1, -1).to(self.device)
        

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
        LOGGER.debug(f"Indices of all parameters: {self.inds_all_params}")

        # Redshift
        self.z0 = self.conf["survey"]["WL"]["z0"]
        self.tomo_z, self.tomo_nz = files.load_redshift_distributions("WL", self.conf)
        if self.conf["analysis"]["modelling"]["WL"]["nla"]["truncate_nz"]:
            self.tomo_z, self.tomo_nz = redshift.get_tomo_nz_arrays_truncated(self.tomo_z, self.tomo_nz, 
                                                                                z_min_quantile=float(self.conf["analysis"]["modelling"]["WL"]["nla"]["z_min_quantile"]), 
                                                                                z_max_quantile=float(self.conf["analysis"]["modelling"]["WL"]["nla"]["z_max_quantile"]))
        self.tomo_z = torch.tensor(self.tomo_z, dtype=torch.float32, device=self.device)
        self.tomo_nz = torch.tensor(self.tomo_nz, dtype=torch.float32, device=self.device)
        
        # Pre-generate samples from latin hypercube
        self.astro_priors = parameters.get_prior_intervals(self.astro_params, conf=self.conf)
        self.astro_samples = prior.sample_astro_parameters_latin_hypercube(
            self.astro_params, 
            seed=self.conf['master_seed']+self.seed, 
            n_examples=num_samples_prior, 
            astro_priors=self.astro_priors)

        LOGGER.info(f"Sampled astrophysical parameters: {self.astro_samples.shape}")
        for i, param_name in enumerate(self.astro_params):
            p_ = self.astro_samples[:, i]
            LOGGER.info(f"   {param_name:>20s}  min={np.min(p_): 8.3f} max={np.max(p_): 8.3f} mean={np.mean(p_): 8.3f}")

        return torch.from_numpy(self.astro_samples).to(self.device)

    def sample_astro_parameters(self, batch_size):

        j = torch.randint(0, self.astro_samples.shape[0], (batch_size,), device=self.device)
        return self.astro_samples[j].squeeze().to(self.device)

    def forward_physics(self, example):

        #
        # Preliminaries
        #

        # unpack the example
        maps, vec_int, cosmo = example
        # gg1, gg2, ga1, ga2, gd1, gd2, ds, dg, qg = maps.unbind(dim=-1)
        gg1, gg2, ga1, ga2, ds, dg = maps.unbind(dim=-1)

        #  we will use gg1 and gg2 as lensing containers
        gg1_tot = gg1
        gg2_tot = gg2

        # get astrophysical parameters
        astro_params = self.sample_astro_parameters(cosmo.shape[0])
        astro_params = torch.atleast_2d(astro_params)
        targets = torch.cat([cosmo, astro_params], dim=1)


        #
        # Convert bary_Mc to log10(bary_Mc)
        #
        targets[:, self.inds_all_params['bary_Mc']] = torch.log10(targets[:, self.inds_all_params['bary_Mc']])
        
        #
        # Linear bias map for galaxy clustering (lenses)
        #
        ng_bar = self.num_gal_gc * self.pixel_area
        ids_bg = [self.inds_astro_params[key] for key in self.conf["analysis"]["params"]["bg"]["linear"]]
        tomo_bg = astro_params[:, ids_bg].unsqueeze(1) # shape (batch_size, 1, n_GC_bins)
        assert dg.shape[-1] == tomo_bg.shape[-1], "The number of bias parameters must match the number of tomographic bins"
        ng_lambda = clustering.galaxy_density_to_count(ng_bar, dg, bg=tomo_bg, qdg=None, qbg=None, mg=None, cg=None, systematics_map=None, mask=None, backend='torch')
        ng_lambda = torch.where(dg == 0, 0, ng_lambda)
        ng = torch.poisson(ng_lambda)
        LOGGER.debug(f'drawn poisson galaxy clustering map ng={ng.shape} min={ng.min():>10.3f} max={ng.max():>10.3f} mean={ng.mean():>10.3f}')

        #
        # Linear bias map for galaxy clustering (sources)
        #
        ng_bar = self.num_gal_wl * self.pixel_area
        ids_bsc = [self.inds_astro_params[key] for key in self.conf["analysis"]["params"]["sc"]]
        tomo_bsc = astro_params[:, ids_bsc].unsqueeze(1) # shape (batch_size, 1, n_WL_bins)
        ns_lambda = clustering.galaxy_density_to_count(ng_bar, ds, bg=tomo_bsc, qdg=None, qbg=None, mg=None, cg=None, systematics_map=None, mask=None, backend='torch')
        ns_lambda = torch.where(ds == 0, 0, ns_lambda)
        ns = torch.poisson(ns_lambda)

        #
        # Add lensing shape noise to g1 and g2
        #
        gg_noise_temp = torch.empty(gg1.shape, dtype=torch.float32, device=self.device)

        for gg in [gg1, gg2]:
            nn.init.trunc_normal_(gg_noise_temp, mean=0.0, std=self.shape_noise_std/math.sqrt(2.), a=-1.0, b=1.0)
            gg_noise_temp.div_(torch.sqrt(ns))
            gg_noise_temp[ns==0] = 0
            gg.add_(gg_noise_temp)
        

        #
        # Add linear intrinsic alignment to g1 and g2
        #
        Aia = astro_params[:, self.inds_astro_params['Aia']]
        nAia = astro_params[:, self.inds_astro_params['n_Aia']]
        tomo_Aia = redshift.get_tomo_amplitudes_vectorized(Aia, nAia, self.tomo_z, self.tomo_nz, self.z0, backend='torch') # shape (batch_size, num_bins)
        tomo_Aia = tomo_Aia.unsqueeze(1) # shape (batch_size, 1, num_bins)
        for ga, gg_tot in zip([ga1, ga2], [gg1_tot, gg2_tot]):
            ga.mul_(tomo_Aia)
            gg_tot.add_(ga)
                           

        # 
        # Wrap up
        # 

        # final report
        LOGGER.debug(f'gg1_tot.shape={gg1_tot.shape}, gg1_tot.dtype={gg1_tot.dtype}')
        LOGGER.debug(f'gg2_tot.shape={gg2_tot.shape}, gg2_tot.dtype={gg2_tot.dtype}')
        LOGGER.debug(f'ns.shape={ns.shape}, ns.dtype={ns.dtype}')
        LOGGER.debug(f'ng.shape={ng.shape}, ng.dtype={ng.dtype}')

        # Stack probes as channels
        inputs = torch.cat([gg1_tot, gg2_tot, ns, ng], dim=-1)
        LOGGER.debug(f'inputs shape={inputs.shape} dtype={inputs.dtype}')

        return inputs, targets

    def apply_scalers(self, inputs, targets):

        inputs = (inputs + self.channel_shift) * self.channel_scale
        targets = (targets + self.targets_shift) * self.targets_scale

        return inputs, targets



    
    def forward(self, example):
        
        inputs, targets = self.forward_physics(example)

        if self.scalers:
            inputs, targets = self.apply_scalers(inputs, targets)

        return inputs, targets

    def unstack_batch_channels(self, inputs):
        """
        Unstack the channels of the input tensor into a list of maps.
        Spin2 maps are stacked into a single tensor.
        Inputs: 
            inputs: tensor of shape (batch_size, num_pixels, num_channels)
        Outputs:
            maps: list of tensors, each tensor is of shape (batch_size, num_pixels) for the scalar maps, and (batch_size, num_pixels, 2) for the spin2 maps
        """

        channel_maps = inputs.unbind(dim=-1)

        # stack spin2 maps into a single tensor
        real_maps = channel_maps[:6]
        imag_maps = channel_maps[6:12]
        scalar_maps = channel_maps[12:18]
        spin2_maps = [torch.stack([r.unsqueeze(1), i.unsqueeze(1)], dim=1) for r, i in zip(real_maps, imag_maps)] # shape (batch_size, 2, num_pixels)
 
        # combine into a single list
        maps = spin2_maps + scalar_maps

        return maps

    