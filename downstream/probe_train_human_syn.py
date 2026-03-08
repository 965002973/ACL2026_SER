
import os
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from torch import nn, optim

from mixdata import load_ssl_features, train_valid_test_dataloader
from model import BaseModel
from utils import train_one_epoch, validate_and_test
import logging

logger = logging.getLogger("MIX_Downstream")


def format_seconds(sec: float):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


@hydra.main(config_path='config', config_name='mix.yaml')
def train_mix(cfg: DictConfig):
    torch.manual_seed(cfg.common.seed)

    label_dict = {
        "human": 0,
        "syn": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    dataset = load_ssl_features(
        cfg.dataset.feat_path,
        label_dict,
    )

    # ✅ 只切 train / val
    train_loader, val_loader, _ = train_valid_test_dataloader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        val_ratio=getattr(cfg.dataset, "val_ratio", 0.11),
        test_ratio=0.0,
        seed=cfg.common.seed,
    )

    model = BaseModel(input_dim=768, output_dim=len(label_dict)).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimization.lr,
        weight_decay=getattr(cfg.optimization, "weight_decay", 1e-4)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.optimization.epoch,
        eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    # ✅ 只保留 Macro-F1 和 Balanced Acc
    best_val_f1 = 0.0
    best_val_balacc = 0.0
    best_epoch = 0

    save_path = os.path.join(str(Path.cwd()), "model_best.pth")

    epoch_times = []
    total_epochs = cfg.optimization.epoch

    for epoch in range(total_epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, optimizer, criterion, train_loader, device
        )
        scheduler.step()

        # validate_and_test 返回 (wa, ua, f1)
        _, val_balacc, val_macro_f1 = validate_and_test(
            model, val_loader, device, num_classes=len(label_dict)
        )

        # ✅ 用 Macro-F1 作为唯一 best 标准
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_balacc = val_balacc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        t1 = time.time()
        epoch_cost = t1 - t0
        epoch_times.append(epoch_cost)

        window = epoch_times[-5:]
        avg_cost = sum(window) / len(window)
        remain = (total_epochs - epoch - 1) * avg_cost

        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"loss {train_loss/len(train_loader):.6f} | "
            f"val MacroF1 {val_macro_f1:.2f} | "
            f"val BalancedAcc {val_balacc:.2f} | "
            f"epoch time {format_seconds(epoch_cost)} | "
            f"ETA {format_seconds(remain)}"
        )

    # ✅ 训练结束：只汇报 best Macro-F1 & Balanced Acc
    model.load_state_dict(torch.load(save_path), strict=True)
    logger.info(
        f"Best epoch {best_epoch+1}: "
        f"best val MacroF1 {best_val_f1:.2f} | "
        f"best val BalancedAcc {best_val_balacc:.2f}"
    )


if __name__ == "__main__":
    train_mix()
