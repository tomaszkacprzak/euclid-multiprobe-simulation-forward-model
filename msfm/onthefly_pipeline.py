# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import warnings
from typing import Union
import webdataset
import torch
from torch.utils.data import DataLoader
from msfm.utils import logger
import glob


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline():
    """
    Sets up a dataset for the onthefly cosmologies.
    """
    def __init__(self, webds_pattern, physics_model, device: torch.device | str, **kwargs):

        self.physics_model = physics_model
        self.device = device

        # get webdataset dataset
        list_files = sorted(glob.glob(webds_pattern))
        LOGGER.info(f"found {len(list_files)} files")

        dataset = (
            webdataset.WebDataset(list_files, shardshuffle=False)
            .decode()
            .to_tuple(
                # "gg.pth",
                # "ga.pth",
                # "gd.pth",
                # "ds.pth",
                # "dg.pth",
                # "qg.pth",
                # "cosmo.pth",
                # "i_sobol.index",
                # "i_signal.index",
                # "n_params.index",
                # "n_pix.index",
                # "n_z_wl.index",
                # "n_z_gc.index",
                "maps_float32.pth",
                "vec_int32.pth",
                "vec_float32.pth",
            )
        )
        
        # get torch DataLoader
        self.loader = DataLoader(dataset, **kwargs)


    def __iter__(self):

        batch_count = 0
        
        for batch in self.loader:

            with torch.profiler.record_function("batch_to_cuda"):
                batch = tuple(tensor.to(self.device) for tensor in batch)

            # add physics augmentations
            with torch.no_grad():

                with torch.profiler.record_function("physics forward model"):
                    inputs, targets = self.physics_model.forward(batch)

            LOGGER.debug(f"Batch {batch_count:>6d} physics shape = {inputs.shape}")

            batch_count += 1

            yield inputs, targets


