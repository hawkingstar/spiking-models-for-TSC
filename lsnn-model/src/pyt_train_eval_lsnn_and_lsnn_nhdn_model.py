#
# This file does the LSNN and LSNN_nhdn model training and evaluation.
#

import _init_paths

import os
import pickle
import torch
import numpy as np
import sys
from PIL import Image  # Import PIL for image handling

from consts.exp_consts import EXC
from src.extract_signals import ExtractSignals
from src.pyt_lsnn_model import LSNN
from src.pyt_lsnn_nhdn_model import LSNN_NHDN
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils import log

from torch.utils.data import DataLoader

from src.patch_dataset import PatchDataset  # Assuming we create a dataset class for patches

class PTTrainEvalModel(object):
    """
    Does PyTorch based training and evaluation of the SNN.
    """
    def __init__(self, dataset, rtc):
        """
        Args:
          dataset <str>: The data on which training and evaluation is to be done.
          rtc <class>: Run Time Constants class.
        """
        self._rtc = rtc
        self._data = dataset
        self._lr = rtc.PYTORCH_LR
        self._batch_size = rtc.BATCH_SIZE
        self._do_normalize = rtc.NORMALIZE_DATASET
        self._test_eval_size = rtc.TEST_EVAL_SIZE

        if rtc.PYTORCH_MODEL_NAME == "LSNN":
            log.INFO("Obtaining LSNN Model with batchsize = %s" % rtc.BATCH_SIZE)
            self._model = LSNN(dataset, rtc)
        if rtc.PYTORCH_MODEL_NAME == "LSNN_NHDN":
            log.INFO("Obtaining LSNN_NHDN with batchsize = %s" % rtc.BATCH_SIZE)
            self._model = LSNN_NHDN(dataset, rtc)

        self._dpu = DataPrepUtils(dataset, rtc)
        self._exs = ExtractSignals(rtc)


def get_batches_of_x_y_from_patches(is_train, patch_dir, num_samples=None, batch_size=16):
    """
    Returns batches of image patches (distorted, clean) instead of LDN signals.

    Args:
        is_train (bool): Load training data if True, else test data.
        patch_dir (str): Path to dataset (distorted & clean patches).
        num_samples (int, optional): Number of samples to load.
        batch_size (int): Batch size for training.

    Yields:
        torch.Tensor: Batch of distorted image patches.
        torch.Tensor: Batch of clean image patches (labels).
    """
    log.INFO("Loading image patches for LSNN training...")

    # Define the transformation manually (resize, convert to tensor)
    def transform(image):
        image = image.resize((32, 32))  # Ensure patches are 32x32
        return torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize the image

    # Select correct dataset path
    if is_train:
        distorted_dir = os.path.join(patch_dir, "train_distorted")
        clean_dir = os.path.join(patch_dir, "train_clean")
    else:
        distorted_dir = os.path.join(patch_dir, "test_distorted")
        clean_dir = os.path.join(patch_dir, "test_clean")

    # Load dataset
    dataset = PatchDataset(distorted_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Yield batches for training
    for distorted_batch, clean_batch in dataloader:
        yield distorted_batch, clean_batch
