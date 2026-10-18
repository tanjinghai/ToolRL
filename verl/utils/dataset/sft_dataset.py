# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frozen Phase 2 messages Parquet adapter, with no silent truncation."""
import pandas as pd
import torch
from torch.utils.data import Dataset
from verl.utils.fs import copy_local_path_from_hdfs
from toolrl_training.sft_data import adapt_sft


class SFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, prompt_key='prompt', prompt_dict_keys=None,
                 response_key='response', response_dict_keys=None, max_length=3072,
                 truncation='error'):
        if truncation != 'error':
            raise ValueError('SFT truncation forbidden: raise max_length or create an audited train-only subset')
        if not isinstance(prompt_key, str) or not isinstance(response_key, str):
            raise ValueError('Use one column name, with optional nested dict keys')
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.dataframe = pd.concat([pd.read_parquet(copy_local_path_from_hdfs(p)) for p in paths], ignore_index=True)
        self.tokenizer, self.max_length = tokenizer, max_length
        def column(name, keys):
            values = self.dataframe[name].tolist()
            for key in keys or []:
                values = [v[key] for v in values]
            return values
        self.prompts = column(prompt_key, prompt_dict_keys)
        self.responses = column(response_key, response_dict_keys)
        # Fail before allocating a model, identifying the exact overlength sample.
        for i in range(len(self)):
            try:
                adapt_sft(self.prompts[i], self.responses[i], tokenizer, max_length)
            except ValueError as exc:
                raise ValueError(f'SFT row {i}: {exc}') from exc

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in
                adapt_sft(self.prompts[item], self.responses[item], self.tokenizer, self.max_length).items()}
