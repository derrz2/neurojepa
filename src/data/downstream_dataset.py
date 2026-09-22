import os
import glob
import re 
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Literal
import torch.nn.functional as F
from .pretrain_dataset import fMRIDataset
import io  
import nibabel as nib

import logging
logger = logging.getLogger("neurojepa")


def _compute_regression_label_stats_from_paths(file_paths, labels_map, subject_id_extractor):
    label_values = []

    for file_path in file_paths:
        subject_id = subject_id_extractor(file_path)
        label_tensor = labels_map.get(subject_id)
        if label_tensor is None:
            continue
        label_values.append(float(label_tensor.view(-1)[0].item()))

    if not label_values:
        raise RuntimeError("No regression labels found while computing training label mean/std.")

    labels = torch.tensor(label_values, dtype=torch.float32)
    mean = float(labels.mean().item())
    std = float(labels.std(unbiased=False).item())
    if std < 1e-8:
        std = 1.0
    return mean, std

