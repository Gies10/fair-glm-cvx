import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import pickle

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from util import Evaluator
from models import define_models
from dataloaders import get_dataset_by_name

save_path = os.path.join(os.getcwd(), 'results')
os.makedirs(save_path, exist_ok=True)

parser = ArgumentParser()
parser.add_argument('--dataset', default='pricingame')
args = parser.parse_args()