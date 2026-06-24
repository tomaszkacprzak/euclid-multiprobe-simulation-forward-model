# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created June 2026
Author: Tomasz Kacprzak
"""
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

    def __init__(self, conf, scalers=False, seed=424344, num_samples_prior=1_000_000, device=None, **kwargs):

        super().__init__()

        self.conf = conf
        self.seed = seed
        self.device = device
        self.astro_samples = self.get_astro_params(num_samples_prior)
        self.shape_noise_std = 0.05
        self.num_gal_wl = torch.from_numpy(np.array(self.conf["survey"]["WL"]["n_gal"])).to(self.device)
        self.num_gal_gc = torch.from_numpy(np.array(self.conf["survey"]["GC"]["n_gal"])).to(self.device)
        self.pixel_area = hp.nside2pixarea(self.conf["analysis"]["n_side"], degrees=True)
        self.sample_uniform_lo = torch.tensor(0., device=self.device, dtype=torch.float32)
        self.sample_uniform_hi = torch.tensor(2 * math.pi, device=self.device, dtype=torch.float32)
        self.num_targets = len(self.all_params)
        self.num_channels = 24
        self.scalers = scalers
        if self.scalers:
            self.set_scalers()
        LOGGER.info(f"Created physics model Linear, num_channels={self.num_channels}, num_targets={self.num_targets}, apply_scalers={self.scalers}, shape_noise_std={self.shape_noise_std}")

    def set_scalers(self):


        sn=0.01
        channel_stds = torch.tensor([ sn, sn,  sn, sn, sn, sn, sn, sn, sn, sn, sn, sn, 6.0742157404e+01, 6.0030681612e+01, 5.8954460206e+01, 5.9754644183e+01, 5.8358482096e+01, 6.0870883217e+01, 1.0521426588e+01, 1.0446036066e+01, 1.0460830854e+01, 1.0405959759e+01, 1.0057117242e+01, 1.0407199295e+01])
        channel_mean = torch.tensor([  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 7.3234141948e+01, 7.2668064310e+01, 7.1583784613e+01, 7.3031979277e+01, 7.1333946636e+01, 7.4637793854e+01, 1.1884575647e+01, 1.1665196170e+01, 1.2027982057e+01, 1.1937562596e+01, 1.1565197428e+01, 1.1962488818e+01]) 
        # shape (1, 1, num_channels)
        self.channel_stds = channel_stds.reshape(1, 1, -1).to(self.device)
        self.channel_mean = channel_mean.reshape(1, 1, -1).to(self.device)


        targets_stds = torch.tensor([7.8334718640e-02,  1.9175510301e-01,  4.5786586271e-03,  5.1527118543e+00,  3.0409003415e-02,  1.8906871713e-01,  4.2933508275e-01,  1.0888102739e+00,  1.6585535022e+00,  2.6264523371e+00,  6.0221754643e-01,  6.7831336488e-01,  6.1259380541e-01,  6.4341219166e-01,  6.3240896897e-01,  6.4718528047e-01,  2.5368841385e-01,  3.0092315697e-01,  3.1627338892e-01,  2.8805443489e-01,  3.0337463573e-01,  3.1135862569e-01])
        targets_mean = torch.tensor([2.6812500358e-01,  8.4843747616e-01,  4.0171875805e-02,  7.1734375000e+01,  1.0009375215e+00, -1.1691046596e+00,  1.3082812500e+01,  4.3125000000e-01, -3.6184755067e-01,  1.7696429292e+00,  1.9737388968e+00,  1.9204200253e+00,  2.0108524859e+00,  1.9878384754e+00,  1.8939310402e+00,  1.9937345639e+00,  1.5251097560e+00,  1.5057281315e+00,  1.4681568265e+00,  1.5181897223e+00,  1.4596939683e+00,  1.5733177662e+00])
        # shape (1, num_targets)
        self.targets_stds = targets_stds.reshape(1, -1).to(self.device)
        self.targets_mean = targets_mean.reshape(1, -1).to(self.device)



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

        return torch.from_numpy(self.astro_samples)

    def sample_astro_parameters(self, batch_size):

        j = torch.randint(0, self.astro_samples.shape[0], (batch_size,))
        return self.astro_samples[j].squeeze()

    def forward_physics(self, example):

        # unpack the example
        maps, vec_int, cosmo = example
        gg1, gg2, ga1, ga2, gd1, gd2, ds, dg, qg = maps.unbind(dim=-1)
        gg = gg1 + 1j*gg2
        ga = ga1 + 1j*ga2
        gd = gd1 + 1j*gd2

        astro_params = self.sample_astro_parameters(cosmo.shape[0]).to(self.device)
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
        with torch.profiler.record_function("linear bias map"):
            ng_bar = self.num_gal_wl * self.pixel_area
            ids_bsc = [self.inds_astro_params[key] for key in self.conf["analysis"]["params"]["sc"]]
            tomo_bsc = astro_params[:, ids_bsc].unsqueeze(1) # shape (batch_size, 1, n_WL_bins)
            ns_lambda = clustering.galaxy_density_to_count(ng_bar, ds, bg=tomo_bsc, qdg=None, qbg=None, mg=None, cg=None, systematics_map=None, mask=None, backend='torch')
            ns_lambda = torch.where(ds == 0, 0, ns_lambda)
            ns = torch.poisson(ns_lambda)

        #
        # Lensing g1 g2 and shape noise
        #
        gg_abs = torch.empty(gg.shape, dtype=torch.float32, device=self.device)
        gg_ang = torch.distributions.Uniform(self.sample_uniform_lo, self.sample_uniform_hi).sample(gg.shape)
        nn.init.trunc_normal_(gg_abs, mean=0.0, std=self.shape_noise_std, a=-1.0, b=1.0)
        gg_noise = gg_abs * torch.exp(1j * gg_ang)
        gg_noise = torch.where(ns>0, gg_noise / torch.sqrt(ns), 0)

        #
        # Linear intrinsic alignment
        #
        Aia = astro_params[:, self.inds_astro_params['Aia']]
        nAia = astro_params[:, self.inds_astro_params['n_Aia']]
        tomo_Aia = redshift.get_tomo_amplitudes_vectorized(Aia, nAia, self.tomo_z, self.tomo_nz, self.z0, backend='torch') # shape (batch_size, num_bins)
        tomo_Aia = tomo_Aia.unsqueeze(1) # shape (batch_size, 1, num_bins)
        ga = ga * tomo_Aia
                           
        # 
        # Total shear map
        # 
        gg_tot = gg + ga + gg_noise 
        gg1_tot = gg_tot.real
        gg2_tot = gg_tot.imag


        # final report
        LOGGER.debug(f'gg1_tot.shape={gg1_tot.shape}, gg1_tot.dtype={gg1_tot.dtype}')
        LOGGER.debug(f'gg2_tot.shape={gg2_tot.shape}, gg2_tot.dtype={gg2_tot.dtype}')
        LOGGER.debug(f'ns.shape={ns.shape}, ns.dtype={ns.dtype}')
        LOGGER.debug(f'ng.shape={ng.shape}, ng.dtype={ng.dtype}')

        # Stack probes as channels
        inputs = torch.cat([gg1_tot, gg2_tot, ns, ng], dim=-1)

        return inputs, targets

    def apply_scalers(self, inputs, targets):

        inputs = (inputs - self.channel_mean) / self.channel_stds
        targets = (targets - self.targets_mean) / self.targets_stds

        return inputs, targets




    def forward(self, example):
        
        inputs, targets = self.forward_physics(example)

        if self.scalers:
            inputs, targets = self.apply_scalers(inputs, targets)

        return inputs, targets



    