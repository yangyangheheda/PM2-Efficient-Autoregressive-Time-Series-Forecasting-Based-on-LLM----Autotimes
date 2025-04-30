import argparse, yaml, os
from autotimes.data import build_dataloader
from autotimes.trainer import AutoTimesTrainer
from autotimes.utils import seed_everything

def main(cfg):
    seed_everything(cfg.get('seed',42))
    train_dl = build_dataloader(split='train', **cfg['data'])
    val_dl   = build_dataloader(split='val', **cfg['data'])
    test_dl  = build_dataloader(split='test', **cfg['data'])
    trainer = AutoTimesTrainer(cfg)
    best_ckpt = trainer.fit(train_dl, val_dl)
    mse, mae = trainer.evaluate(test_dl, best_ckpt)
    with open(os.path.join(cfg['train']['save_dir'], 'result.txt'), 'w') as f:
        f.write(f"MSE={mse:.6f} MAE={mae:.6f}\n")
    print('Test MSE', mse, 'MAE', mae)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c','--config', required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if 'inherit' in cfg:
        import pathlib
        base = pathlib.Path(args.config).with_name(cfg['inherit'])
        with open(base) as bf:
            base_cfg=yaml.safe_load(bf)
        base_cfg.update(cfg)
        cfg=base_cfg
    main(cfg)
