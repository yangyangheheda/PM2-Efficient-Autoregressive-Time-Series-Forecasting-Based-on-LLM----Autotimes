import pandas as pd, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def build_ts_embed(dates, llm_name='gpt2', device='cpu', eos_token='<|endoftext|>'):
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
    model.eval()
    texts = [f'This is the series from {str(a)} to {str(b)} <EOS>' for a,b in dates]
    with torch.inference_mode():
        inputs = tokenizer(texts, padding=True, return_tensors='pt').to(device)
        outs = model(**inputs, output_hidden_states=True).hidden_states[-1]
        mask = inputs['attention_mask'].sum(1)-1
        eos_vec = outs[torch.arange(outs.size(0)), mask]
    return eos_vec.cpu()

class SegmentDataset(Dataset):
    def __init__(self, csv_path, seg_len=96, win_tokens=10, split='train', num_vars=1, llm_name='gpt2'):
        df = pd.read_csv(csv_path)
        if 'OT' in df.columns:
            vals = df['OT'].values.astype('float32')
        else:
            vals = df.iloc[:,1].values.astype('float32')
        dates = pd.to_datetime(df.iloc[:,0])
        total = len(vals)//seg_len
        idx_tr = int(total*0.7)
        idx_v  = int(total*0.8)
        if split=='train':
            start,end = 0,idx_tr
        elif split=='val':
            start,end = idx_tr,idx_v
        else:
            start,end = idx_v,total
        segs = torch.tensor(vals[:total*seg_len]).view(total,seg_len)
        self.segs = segs[start:end]
        self.dates = dates.values[:total*seg_len].reshape(total,seg_len)[start:end][:,0]
        self.win_tokens=win_tokens
        self.seg_len=seg_len
        self.llm_name=llm_name
        cache = Path(csv_path).with_suffix(f'.{split}.ts.pt')
        if cache.exists():
            self.ts = torch.load(cache)
        else:
            pairs = [(d, d+timedelta(hours=seg_len-1)) for d in self.dates]
            self.ts = build_ts_embed(pairs, llm_name=llm_name)
            torch.save(self.ts, cache)

    def __len__(self):
        return len(self.segs)-self.win_tokens-1

    def __getitem__(self, idx):
        x = self.segs[idx:idx+self.win_tokens]
        ts = self.ts[idx:idx+self.win_tokens]
        y = self.segs[idx+self.win_tokens]
        return x, ts, y

def build_dataloader(name, path, batch, win_seg, seg_len=96, split='train', num_workers=4, llm_name='gpt2'):
    ds = SegmentDataset(path, seg_len, win_seg, split, llm_name=llm_name)
    return DataLoader(ds, batch_size=batch, shuffle=(split=='train'), num_workers=num_workers)
