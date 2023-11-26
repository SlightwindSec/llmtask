import os

import pandas as pd

from _subject_names import *


class TaskGenerator:
    def __init__(self, dataset: str, max_shot = 5, tokenizer = None, max_tokens = 2048, start = 0, end = None, pbar: bool = True) -> None:
        self.dataset = dataset
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(current_path, 'datasets', str(dataset))
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f'Dataset dir {self.dataset_dir} does not exist.')
        

    def _load_data(self, dataset):
        pass