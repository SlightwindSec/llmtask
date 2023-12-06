import os

import pandas as pd
import tqdm

from ._subject_names import SUPPORTED_DATASETS, SUBJECT_NAMES_MAP
from ._utils import gen_prompt, format_example, parse_answer
from .error import FeedbackNotCalledError



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
        
        self.answer_log = kwargs.get('csv')
        
        
        self.subject_idx = 0
        self.problem_idx = 0
        
        self.infer_result = []
        
        self._load_cache()

        self.total_task_num = self._get_total_task_num()
        
        self.prompt_prefix = None
        
        self.subject_dev_df = self._load_subject_df(dev=True)
        self.subject_val_df = self._load_subject_df()
        
        self.pbar = None
        if kwargs.get('pbar') is not False:
            self.pbar = tqdm.tqdm(total=self.total_task_num, initial=self.start)
        
        self.truth = None
        self.feedback_flag = False
        

    def _load_cache(self):
        if not self.log_dir:
            return
        cache_file = os.path.join(self.log_dir, f"{self.dataset}_answers.csv")
        if not os.path.exists(cache_file):
            return
        with open(cache_file, "r") as f:
            c = f.read().strip().split("\n")
        if not c or len(c) <= 1:
            return
        self.start = int(c[-1].split(", ")[0])


    def _read_csv(self, csv_path, dev=False):
        if self.dataset == 'ceval':
            drop_col = ['id', 'explanation'] if dev else ['id']
            df = pd.read_csv(csv_path).drop(drop_col, axis=1)
        elif self.dataset == 'mmlu':
            df = pd.read_csv(csv_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
        return df


    def _load_subject_df(self, dev=False):
        _sub_dir = "dev" if dev else self.sub_dir
        subject_name = self.subject_names[self.subject_idx]
        _tasks_dir = os.path.join(self.dataset_dir, _sub_dir)
        _csv_path = os.path.join(_tasks_dir, f"{subject_name}_{_sub_dir}.csv")
        return self._read_csv(_csv_path, dev)
    
    
    def _construct_prompt_prefix(self):
        '''Construct a N-Shot prompt prefix from the `dev` subdirectory'''
        subject_name = self.subject_names[self.subject_idx]
        prompt_prefix = gen_prompt(self.subject_dev_df, subject_name, self.max_shot)
        return prompt_prefix
    
    
    def _get_total_task_num(self):
        '''returns the total number of tasks'''
        _total_task_num = 0
        _tasks_dir = os.path.join(self.dataset_dir, self.sub_dir)
        for i, subject_name in enumerate(self.subject_names):
            _csv_path = os.path.join(_tasks_dir, f"{subject_name}_{self.sub_dir}.csv")
            _tmp_df = self._read_csv(_csv_path)
            if self.start >= _total_task_num and self.start < (_total_task_num + _tmp_df.shape[0]):
                self.subject_idx = i
                self.problem_idx = self.start - _total_task_num
            _total_task_num += _tmp_df.shape[0]
        return _total_task_num
    
    
    def feedback(self, answer):
        self.feedback_flag = False
        model_choice = parse_answer(answer)
        subject_name = self.subject_names[self.subject_idx]
        result = {"subject": subject_name, "problem_idx": self.problem_idx, "truth": self.truth, "guess": model_choice}

        if self.answer_log:
            is_exist = os.path.exists(self.answer_log)
            if is_exist:
                with open(self.answer_log, 'r') as f:
                    header = f.read().split('\n')[0]
            if not is_exist or not header:
                with open(self.answer_log, 'a') as f:
                    f.write("subject, problem_idx, truth, guess\n")
            with open(self.answer_log, 'a') as f:
                f.write(f"{subject_name}, {self.problem_idx}, {self.truth}, {model_choice}\n")
        
        self.infer_result.append(result)
        
        self.problem_idx += 1
        if self.problem_idx >= self.subject_val_df.shape[0]:
            self.subject_idx += 1
            self.problem_idx = 0
            if self.subject_idx < len(self.subject_names):
                self.subject_dev_df = self._load_subject_df(dev=True)
                self.subject_val_df = self._load_subject_df()
            
        if self.pbar:
            self.pbar.update(1)
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        if self.feedback_flag:
            if self.pbar: self.pbar.close()
            raise FeedbackNotCalledError
        self.feedback_flag = True
        
        if self.subject_idx >= len(self.subject_names):
            if self.pbar: self.pbar.close()
            raise StopIteration
        
        prompt_prefix = self._construct_prompt_prefix()
        prompt_problem = format_example(self.subject_val_df, self.problem_idx, False)
        self.truth = self.subject_val_df.answer.iloc[self.problem_idx]
        
        prompt = prompt_prefix + prompt_problem

        return prompt

    def summary(self):
        subjects_acc = {}
        category_acc = {}
        for res in self.infer_result:
            subject_name, _, x, y = res
            if subject_name in subjects_acc:
                subjects_acc[subject_name] += int(x == y)
            else:
                subjects_acc[subject_name]  = int(x == y)
            
    