import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, Subset

logger = logging.getLogger(__name__)

def load_dataset(data_path, labels=None, min_length=3, max_length=None):
    sizes = []
    offsets = []
    emo_labels = []

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
            lbl = None if labels is None else next(lbl_f).rstrip().split()[1]  # only emo is needed
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels


class SpeechDataset(Dataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        labels=None,
        domain_ids=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()
        
        self.feats = feats
        self.sizes = sizes  # length of each sample
        self.offsets = offsets  # offset of each sample

        self.labels = labels
        self.domain_ids = domain_ids
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        # res = {"id": index, "feats": feats}
        # if self.labels is not None:
        #     res["target"] = self.labels[index]

        # return res
        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]
        if self.domain_ids is not None:
            res["domain_id"] = self.domain_ids[index]   # ✅ 新增
        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

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
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

# def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):

#     data, sizes, offsets, labels = load_dataset(feature_path, labels='emo', min_length=1, max_length=max_speech_seq_len)
#     labels = [ label_dict[elem] for elem in labels ]
    
#     num = len(labels)
#     iemocap_data = {
#         "feats": data,
#         "sizes": sizes,
#         "offsets": offsets,
#         "labels": labels,
#         "num": num
#     } 

#     return iemocap_data

class DomainBalancedBatchSampler(Sampler):
    """
    每个 batch 保证各 domain 均衡采样
    domain_ids: list[int], len=N
    domains: 可选，默认从 domain_ids 自动推断
    """
    def __init__(self, domain_ids, batch_size, seed=1, drop_last=True, domains=None):
        self.domain_ids = np.array(domain_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)

        if domains is None:
            self.domains = sorted(list(set(self.domain_ids.tolist())))
        else:
            self.domains = domains

        self.num_domains = len(self.domains)
        assert batch_size % self.num_domains == 0, \
            f"batch_size({batch_size}) 必须能被域数({self.num_domains})整除"
        self.per_domain = batch_size // self.num_domains

        # 每个域的 index 池
        self.domain2indices = {
            d: np.where(self.domain_ids == d)[0].tolist()
            for d in self.domains
        }

    def __iter__(self):
        # 每轮都 shuffle 各域池
        pools = {}
        for d, idxs in self.domain2indices.items():
            idxs = idxs.copy()
            self.rng.shuffle(idxs)
            pools[d] = idxs

        # 取最短的域来决定 batch 数（避免某域被过采样太多）
        min_len = min(len(pools[d]) for d in self.domains)
        num_batches = min_len // self.per_domain
        if not self.drop_last and min_len % self.per_domain != 0:
            num_batches += 1

        for b in range(num_batches):
            batch = []
            for d in self.domains:
                start = b * self.per_domain
                end = start + self.per_domain
                chunk = pools[d][start:end]

                # 不够就跳过（drop_last）
                if len(chunk) < self.per_domain:
                    continue
                batch.extend(chunk)

            if len(batch) == self.batch_size:
                self.rng.shuffle(batch)
                yield batch

    def __len__(self):
        min_len = min(len(v) for v in self.domain2indices.values())
        num_batches = min_len // self.per_domain
        if not self.drop_last and min_len % self.per_domain != 0:
            num_batches += 1
        return num_batches
    
def load_ssl_features(feature_path, label_dict, domain_ids=None, max_speech_seq_len=None):

    data, sizes, offsets, labels = load_dataset(
        feature_path,
        labels='emo',
        min_length=1,
        max_length=max_speech_seq_len
    )
    labels = [label_dict[elem] for elem in labels]

    speech_dataset = SpeechDataset(
        feats=data,
        sizes=sizes,
        offsets=offsets,
        labels=labels,
        domain_ids=domain_ids, 
    )

    num = len(labels)
    mixed_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "num": num,
        "dataset": speech_dataset,   # ✅ 新增
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
    full_dataset: SpeechDataset = data["dataset"]
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

def train_valid_test_dataloader_balanced(
    data,
    batch_size,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=1,
):
    """
    混合训练 + Domain-balanced batch
    """
    full_dataset: SpeechDataset = data["dataset"]
    num = len(full_dataset)

    assert hasattr(full_dataset, "domain_ids") and full_dataset.domain_ids is not None, \
        "SpeechDataset 里必须提供 domain_ids 才能 balanced 采样"

    domain_ids = np.array(full_dataset.domain_ids)

    # 1) 可复现 shuffle
    rng = np.random.RandomState(seed)
    indices = np.arange(num)
    rng.shuffle(indices)

    n_test = int(num * test_ratio)
    n_val  = int(num * val_ratio)

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    # 2) 给 train/val 构造 balanced batch_sampler
    train_domain_ids = domain_ids[train_idx].tolist()
    val_domain_ids   = domain_ids[val_idx].tolist()

    train_batch_sampler = DomainBalancedBatchSampler(
        train_domain_ids, batch_size=batch_size, seed=seed, drop_last=True
    )
    val_batch_sampler = DomainBalancedBatchSampler(
        val_domain_ids, batch_size=batch_size, seed=seed, drop_last=False
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,   # ✅ 关键：用 batch_sampler
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_sampler=val_batch_sampler,     # ✅ val 也 balanced
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=full_dataset.collator,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
