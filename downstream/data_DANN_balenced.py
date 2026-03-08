import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, Subset

from itertools import cycle

logger = logging.getLogger(__name__)


def load_DANN_dataset(data_path, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []
    domain_labels = []

    npy_data = np.load(data_path + ".npy")

    offset = 0
    skipped = 0

    if not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            # emotion label and domain label (synthesized model name)
            _, lbl_emo, lbl_domain = (None, None, None) if labels is None else next(lbl_f).rstrip().split()
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)
                if lbl_emo is not None:
                    emo_labels.append(lbl_emo)
                    domain_labels.append(lbl_domain)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels, domain_labels

class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        emo_labels=None,
        domain_labels=None,
        domain_ids=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample

        self.emo_labels = emo_labels
        self.domain_labels = domain_labels
        self.domain_ids = domain_ids
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.emo_labels is not None:
            res["target_emo"] = self.emo_labels[index]
        if self.domain_labels is not None:
            res["target_domain"] = self.domain_labels[index]
        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        emo_labels = torch.tensor([s["target_emo"] for s in samples]) if samples[0]["target_emo"] is not None else None
        domain_labels = torch.tensor([s["target_domain"] for s in samples]) if samples[0]["target_domain"] is not None else None

        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )

        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask
            },
            "emo_labels": emo_labels,
            "domain_labels": domain_labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    

def load_DANN_features(feature_path, emo_label_dict, domain_label_dict, domain_ids=None, max_speech_seq_len=None):

    data, sizes, offsets, emo_labels, domain_labels = load_DANN_dataset(
        feature_path,
        labels='dann',
        min_length=1,
        max_length=max_speech_seq_len
    )

    emo_labels = [emo_label_dict[elem] for elem in emo_labels]
    domain_labels = [domain_label_dict[elem] for elem in domain_labels]

    num = len(emo_labels)
    mixed_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "emo_labels": emo_labels,
        "domain_labels": domain_labels,
        "num": num,
    }

    return mixed_data


def train_valid_test_dataloader(
    data,
    batch_size,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=1,
):
    """
    通用随机划分 dataloader：
    - 不依赖 Session / n_samples
    - 适用于混合后的大数据集
    """
    # full_dataset: SpeechDataset = data
    full_dataset = SpeechDataset(
        feats=data["feats"],
        sizes=data["sizes"],
        offsets=data["offsets"],
        emo_labels=data["emo_labels"],
        domain_labels=data["domain_labels"]
    )
    num = len(full_dataset)

    # 可复现 shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(num)
    rng.shuffle(indices)

    n_test = int(num * test_ratio)
    n_val = int(num * val_ratio)

    test_idx = indices[:n_test]
    val_idx  = indices[n_test:n_test + n_val]
    train_idx= indices[n_test + n_val:]

    # 用 Subset 切分
    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set   = torch.utils.data.Subset(full_dataset, val_idx)
    test_set  = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader, test_loader



from itertools import cycle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def _list_collate(samples):
    # 关键：返回 list，不做 torch 默认 collate（否则可变长会炸）
    return samples

def train_valid_dataloader_domain_balanced(
    data,
    batch_size,
    val_ratio=0.1,
    seed=1,
    num_workers=4,
    domain_major_id=0,  # Human
    domain_minor_id=1,  # Synthetic
):
    """
    返回：
      real_loader: 每次返回 half-batch 的样本 list（未 padding）
      syn_loader : 每次返回 half-batch 的样本 list（未 padding）
      val_loader : 正常 collate（已 padding）
      full_dataset: 用于 collator（训练时把 real+syn 合并后再 pad）
    """

    full_dataset = SpeechDataset(
        feats=data["feats"],
        sizes=data["sizes"],
        offsets=data["offsets"],
        emo_labels=data["emo_labels"],
        domain_labels=data["domain_labels"]
    )
    num = len(full_dataset)

    rng = np.random.RandomState(seed)
    indices = np.arange(num)
    rng.shuffle(indices)

    n_val = int(num * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # ---- 从 train_idx 中再按 domain 分组 ----
    # 注意：domain_labels 是 list，与 full_dataset 的 index 对齐
    train_domains = np.array([full_dataset.domain_labels[i] for i in train_idx])

    real_idx = train_idx[train_domains == 0]
    syn_idx  = train_idx[train_domains != 0]

    if len(real_idx) == 0 or len(syn_idx) == 0:
        raise ValueError(f"Empty domain split: real={len(real_idx)}, syn={len(syn_idx)}")

    if batch_size % 2 != 0:
        raise ValueError("batch_size 必须是偶数，以实现 1:1 domain batch")
    half = batch_size // 2

    real_set = Subset(full_dataset, real_idx.tolist())
    syn_set  = Subset(full_dataset, syn_idx.tolist())
    val_set  = Subset(full_dataset, val_idx.tolist())

    # 训练 loader：不做 collate，只吐出 list[dict]
    real_loader = DataLoader(
        real_set,
        batch_size=half,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_list_collate,
    )
    syn_loader = DataLoader(
        syn_set,
        batch_size=half,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_list_collate,
    )

    # 验证 loader：保持你原来的 collator（padding）
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=full_dataset.collator,
    )

    return real_loader, syn_loader, val_loader, full_dataset
