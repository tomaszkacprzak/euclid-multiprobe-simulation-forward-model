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
from torch.utils.data import DataLoader
from msfm.utils import logger
from msfm.utils.torch_datasets import ONTHEFLY_TUPLE_KEYS, make_msfm_collate_fn
import glob


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline():
    """
    Sets up a dataset for the onthefly cosmologies.
    """
        

    def get_dataset(
        self,
        webds_pattern: str,
    ):
        """Builds the dataset from the given file name pattern and performance related parameters.

        Args:
            webds_pattern (str): Glob pattern of the webdataset tar files.
            batch_size (int): Local batch size. 
        Returns:
            loader (torch.utils.data.DataLoader): A data loader that yields batches of data.
        """

        list_files = sorted(glob.glob(webds_pattern))
        LOGGER.info(f"list_files = {list_files}")

        dataset = (
            webdataset.WebDataset(list_files, shardshuffle=False)
            .shuffle(1000)
            .decode()
            .to_tuple(
                "gg.pth",
                "ga.pth",
                "gd.pth",
                "ds.pth",
                "dg.pth",
                "qg.pth",
                "cosmo.pth",
                "i_sobol.index",
                "i_signal.index",
                "n_params.index",
                "n_pix.index",
                "n_z_wl.index",
                "n_z_gc.index",
            )
        )

        return dataset

    def get_loader(
        self,
        webds_pattern: str,
        batch_size: int,
        float_dtype=None,
    ):
        """Builds the data loader from the given dataset and performance related parameters.

        The custom collate function returns dictionary batches, casts floating
        arrays to ``torch.float32`` by default, and casts metadata indices to
        ``torch.int64``. Pass ``float_dtype`` to intentionally override the
        floating-point dtype.
        """

        dataset = self.get_dataset(webds_pattern)
        if float_dtype is None:
            collate_fn = make_msfm_collate_fn(tuple_keys=ONTHEFLY_TUPLE_KEYS)
        else:
            collate_fn = make_msfm_collate_fn(float_dtype=float_dtype, tuple_keys=ONTHEFLY_TUPLE_KEYS)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, collate_fn=collate_fn)

        return loader
