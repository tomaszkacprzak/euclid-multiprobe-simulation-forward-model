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
    def __init__(self, webds_pattern, physics_model, smoothing_model=None, device="cuda", **kwargs):

        self.physics_model = physics_model
        self.smoothing_model = smoothing_model
        self.device = device

        # get webdataset dataset
        list_files = sorted(glob.glob(webds_pattern))
        LOGGER.info(f"found {len(list_files)} files")

        dataset = (
            webdataset.WebDataset(
                list_files, 
                shardshuffle=1000,
            )
            .decode()
            .to_tuple(
                "maps_float32.pth",
                "vec_int32.pth",
                "vec_float32.pth",
            )
        )
        
        # get torch DataLoader
        # self.loader = DataLoader(dataset, **kwargs)
        self.loader = webdataset.WebLoader(
                        dataset,
                        pin_memory=True,
                        **kwargs
                      )


        # test and get the number of pixels
        for inputs, targets in self.__iter__():
            
            self.num_pixels = inputs.shape[1]
            self.num_channels = inputs.shape[2]
            self.num_targets = targets.shape[1]
            break

        LOGGER.info(f"Created OntheflyPipeline with num_pixels={self.num_pixels}, num_channels={self.num_channels}, num_targets={self.num_targets}")


    def __iter__(self):

        batch_count = 0
        
        for batch in self.loader:

            with torch.profiler.record_function("batch_to_cuda"):
                batch = tuple(tensor.to(self.device, non_blocking=True) for tensor in batch)

            # maps, vec_int, cosmo = batch
        #    "vec_int32.pth": torch.from_numpy(np.array([i_sobol, i_signal, nside, nside_down]).astype(np.int32)), 
            # batch_sobol_ids = ' '.join([f'{v:>5d}' for v in vec_int[:,1]])
            # LOGGER.info(f"Batch {batch_count:>6d} sobol ids: {batch_sobol_ids}")

            # add physics augmentations
            with torch.no_grad():

                with torch.profiler.record_function("physics forward model"):
                    inputs, targets = self.physics_model.forward(batch)
                    if self.smoothing_model is not None:
                        inputs = self.smoothing_model(inputs)

            LOGGER.debug(f"Batch {batch_count:>6d} physics shape = {inputs.shape}")

            batch_count += 1

            yield inputs, targets


