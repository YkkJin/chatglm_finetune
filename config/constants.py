import os 
from pathlib import Path



ROOT_DIR = Path(os.path.dirname(__file__)).parent

DATA_DIR = os.path.join(ROOT_DIR,'data')

CONFIG_DIR = os.path.dirname(__file__)

