import os
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from torch import nn, optim

from data_DANN_balenced import load_DANN_features, train_valid_dataloader_domain_balanced
from model import EmotionDANN
from utils_DANN_balenced import train_one_epoch, validate_and_test
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

@hydra.main(config_path='config', config_name='DANN.yaml')
def train_DANN(cfg: DictConfig):
    torch.manual_seed(cfg.common.seed)

    emo_label_dict = {
        "ang": 0,
        "hap": 1,
        "neu": 2,
        "sad": 3,
        "fea": 4,
        "dis": 5,
        "sur": 6,
    }

    syn_dict = {
        "Human": 0,
        "COSY": 1,
        "KIMI": 1,
        "GLM": 1,
        "INDEX": 1,
        "4o_TTS": 1,
        "4o_Audio": 1,
    } 
    vocoder_dict = {  # 0:human, 1:HIFI-GAN, 2:BigVGAN, 3:openai
        "Human": 0,
        "COSY": 1,
        "KIMI": 2,
        "GLM": 1,
        "INDEX": 2,
        "4o_TTS": 3,
        "4o_Audio": 3,
    }
    model_dict = {
        "Human": 0,
        "COSY": 1,
        "KIMI": 2,
        "GLM": 3,
        "INDEX": 4,
        "4o_TTS": 5,
        "4o_Audio": 6,
    }

    # 选择使用哪个 domain label 进行 Domain Adversarial Training
    # Choices: [syn_dict, vocoder_dict, model_dict]
    domain_label_dict = syn_dict 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    dataset = load_DANN_features(
        cfg.dataset.feat_path,
        emo_label_dict,
        domain_label_dict,
    )
    from data_DANN_balenced import train_valid_dataloader_domain_balanced

    real_loader, syn_loader, val_loader, full_dataset = train_valid_dataloader_domain_balanced(
        dataset,
        batch_size=cfg.dataset.batch_size,
        val_ratio=getattr(cfg.dataset, "val_ratio", 0.11),
        seed=cfg.common.seed,
        num_workers=0,
        domain_major_id=0,  # Human
        domain_minor_id=1,  # Synthetic
    )
    # train_loader, val_loader, _ = train_valid_test_dataloader(
    #     dataset,
    #     batch_size=cfg.dataset.batch_size,
    #     val_ratio=getattr(cfg.dataset, "val_ratio", 0.11),
    #     test_ratio=0.0,  
    #     seed=cfg.common.seed,
    # )

    model = EmotionDANN(
        input_dim=768, 
        num_emotions=len(set(emo_label_dict.values())), 
        num_domains=len(set(domain_label_dict.values()))
    ).to(device)
    # model = ModelWithProj(output_dim=len(label_dict)).to(device)

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

    best_val_wa, best_val_ua, best_val_f1, best_epoch = 0.0, 0.0, 0.0, 0
    save_path = os.path.join(str(Path.cwd()), "model_best_balenced.pth")

    epoch_times = []
    total_epochs = cfg.optimization.epoch

    for epoch in range(total_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            real_loader,
            syn_loader,
            full_dataset.collator,  # ✅ 关键：传 collator
            epoch,
            total_epochs,
            max_alpha=cfg.optimization.max_alpha,
            device=device,
        )
        # train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch, total_epochs, device)
        scheduler.step()

        # 只在 val 上评估
        val_wa, val_ua, val_f1 = validate_and_test(
            model, val_loader, device, num_classes=len(emo_label_dict)
        )

        if val_wa > best_val_wa:
            best_val_wa, best_val_ua, best_val_f1, best_epoch = val_wa, val_ua, val_f1, epoch
            torch.save(model.state_dict(), save_path)

        t1 = time.time()
        epoch_cost = t1 - t0
        epoch_times.append(epoch_cost)

        window = epoch_times[-5:]
        avg_cost = sum(window) / len(window)
        remain = (total_epochs - epoch - 1) * avg_cost

        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"loss {train_loss/len(real_loader):.6f} | "
            f"val WA {val_wa:.2f} UA {val_ua:.2f} F1 {val_f1:.2f} | "
            f"epoch time {format_seconds(epoch_cost)} | "
            f"ETA {format_seconds(remain)}"
        )

    # 训练结束：只加载 best 并汇报 best val（不测 test）
    model.load_state_dict(torch.load(save_path), strict=True)
    logger.info(
        f"Best epoch {best_epoch+1}: "
        f"best val WA {best_val_wa:.2f} UA {best_val_ua:.2f} F1 {best_val_f1:.2f}"
    )

if __name__ == "__main__":
    train_DANN()


