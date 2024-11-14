import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

class DatasetLoader(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self._data = HFDataset.from_pandas(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        goal = self._data[idx]['goal']
        suffixes = eval(self._data[idx]['response'])
        
        # Delimiter between each suffix so I can access the output easier later on. 
        # Probably not the best way to do this but it works for now.
        suffix_text = ""
        for suffix in suffixes:
            suffix_text += f' {suffix} |'
        input_text = goal + ' |' + suffix_text

        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        labels = inputs.input_ids.clone()

        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

    def data_collator(self, features):
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        labels = torch.stack([f['labels'] for f in features])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}