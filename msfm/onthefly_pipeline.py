# Copyright (C) 2026 FHNW, Institute for Data Science

"""
Created February 2026
Author: Tomasz Kacprzak

This file is loosely based off
grid_pipeline.py by Arne Thomsen
"""

import tensorflow as tf
import warnings
from typing import Union

from msfm.utils import logger, tfrecords, parameters
from msfm.utils.base_pipeline import MSFMpipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


class OntheflyPipeline():
    """
    Sets up a dataset for the onthefly cosmologies.
    """

    def __init__(
        self
    ):
        pass

        # TODO: implement this

    def get_dset(
        self,
        webds_pattern: str,
        local_batch_size: int,
    ) -> tf.data.Dataset:
        """Builds the dataset from the given file name pattern and performance related parameters.

        Args:
            webds_pattern (str): Glob pattern of the webdataset tar files.
            local_batch_size (int): Local batch size. 
        Returns:
            pytorch data loader
        """

        # TODO: implement this
        loader = ...

        

        return loader

