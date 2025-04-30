import torch, os, math
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from autotimes.model import AutoTimesModel
from peft import get_peft_model, LoraConfig

class AutoTimesTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        llm_cfg = cfg['llm']
        self.model = AutoTimesModel(llm_cfg['name'], cfg['seg_len'])
        if llm_cfg.get('lora', False):
            peft_cfg = LoraConfig(r=llm_cfg.get('lora_rank',8),
                                  lora_alpha=llm_cfg.get('lora_alpha',16),
                                  target_modules=["c_attn","c_proj"],
                                  task_type="CAUSAL_LM")
            self.model.llm = get_peft_model(self.model.llm, peft_cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.crit = nn.MSELoss()
        self.optim = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                           lr=cfg['train']['lr'])

    def fit(self, train_dl, val_dl):
        best = math.inf
        save_dir = Path(self.cfg['train']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(1, self.cfg['train']['epochs']+1):
            self.model.train()
            tot=0
            for x,ts,y in tqdm(train_dl, desc=f'Epoch {ep}'):
                x,ts,y = x.to(self.device), ts.to(self.device), y.to(self.device)
                preds = self.model(x, ts)
                loss = self.crit(preds, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                tot += loss.item()*x.size(0)
            val = self.evaluate(val_dl)
            if val < best:
                best=val
                torch.save(self.model.state_dict(), save_dir/'best.pt')
        return save_dir/'best.pt'

    def evaluate(self, dl, ckpt=None):
        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        tot=0
        mae=0
        with torch.no_grad():
            for x,ts,y in dl:
                x,ts,y = x.to(self.device),ts.to(self.device),y.to(self.device)
                preds = self.model(x, ts)
                tot += self.crit(preds,y).item()*x.size(0)
                mae += torch.mean(torch.abs(preds-y)).item()*x.size(0)
        return tot/len(dl.dataset), mae/len(dl.dataset)
