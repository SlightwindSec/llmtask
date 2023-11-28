import os

import pandas as pd
import tqdm

from ._subject_names import SUPPORTED_DATASETS, SUBJECT_NAMES_MAP


class TaskGenerator:
    def __init__(self, dataset: str, max_shot = 5, tokenizer = None, max_tokens = 2048, **kwargs) -> None:
        '''
        TaskGenerator
        Parameters:
            dataset: string, "mmlu" or "ceval", means MMLU dataset or C-Eval dataset.
            max_shot: int, Few-shot (or zero-shot) prompts, maximum number of question-answer pairs to show before the actual question.
            tokenizer: (optional) transformers tokenizer, 
            max_tokens: (optional) int, maximum number of tokens to generate, only effective when configuring the tokenizer.
            start: (optional) int, start position of tasks, default 0.
            end: (optional) int, end position of tasks, default is the lastest task.
        '''
        # the dataset is supported
        assert dataset.strip().lower() in SUPPORTED_DATASETS, f"Invalid dataset name: {dataset}, expected {', '.join(SUPPORTED_DATASETS)}"
        
        self.dataset = dataset
        self.max_shot = max_shot

        # check dataset directory (already downloaded when installing this package)
        self.dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', str(dataset))
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f'Dataset dir {self.dataset_dir} does not exist.')
        
        # sub directory of the dataset, 'dev', 'val', 'test'
        self.sub_dir = "val" if kwargs.get('sub_dir') is None else kwargs.get('sub_dir')
        
        self.start = 0 if kwargs.get('start') is None else kwargs.get('start')
        self.end = kwargs.get('end')
        self.subject_names = SUBJECT_NAMES_MAP[self.dataset]
        
        self.log_dir = kwargs.get('log_dir')
        if self.log_dir and not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        self.total_task_num = self._get_total_task_num()
        
        self.pbar = None
        if kwargs.get('pbar') is not False:
            self.pbar = tqdm.tqdm(total=self.total_task_num, initial=self.start)

    def _read_csv(self, csv_path, dev=False):
        if self.dataset == 'ceval':
            drop_col = ['id', 'explanation'] if dev else ['id']
            df = pd.read_csv(csv_path).drop(drop_col)
        elif self.dataset == 'mmlu':
            df = pd.read_csv(csv_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
        return df
    
    def _get_total_task_num(self):
        '''returns the total number of tasks'''
        _total_task_num = 0
        _tasks_dir = os.path.join(self.dataset_dir, self.sub_dir)
        for subject_name in self.subject_names:
            _csv_path = os.path.join(_tasks_dir, f"{subject_name}_{self.sub_dir}.csv")
            _tmp_df = self._read_csv(_csv_path)
            _total_task_num += _tmp_df.shape[0]
        return _total_task_num
    
    def __next__(self):
        pass