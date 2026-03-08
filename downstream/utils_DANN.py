import torch
import numpy as np

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, total_epochs, max_alpha, device):
    model.train()
    train_loss = 0
    train_emo_loss = 0
    train_domain_loss = 0
    # Lambda (alpha) 预热。 刚开始训练时，特征提取器不好，如果对抗太强，模型会崩溃。
    # 动态调整 alpha (对抗强度)：从 0 -> 1
    # 刚开始让特征提取器先学怎么分情绪，慢慢再加入领域混淆
    p = epoch / total_epochs
    alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)* max_alpha
    
    for batch in train_loader:
        ids, net_input, emo_labels, domain_labels = batch["id"], batch["net_input"], batch["emo_labels"], batch["domain_labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)

        speech_padding_mask = speech_padding_mask.to(device)
        emo_labels = emo_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        optimizer.zero_grad()

        # valid_lens = (~speech_padding_mask).sum(dim=1)   # 假设 True=pad
        # print("valid_lens min/mean/max:", valid_lens.min().item(), valid_lens.float().mean().item(), valid_lens.max().item())
        # print("mask true ratio:", speech_padding_mask.float().mean().item())

        emotion_pred, domain_pred = model(feats, speech_padding_mask, alpha=alpha)
        
        loss_emo = criterion(emotion_pred, emo_labels.long())
        loss_label = criterion(domain_pred, domain_labels.long())
        
        # 权重可以试着调一调
        loss = 1.0 * loss_emo + 1.0 * loss_label

        train_loss += loss.item()
        train_emo_loss += loss_emo.item()
        train_domain_loss += loss_label.item()
        
        loss.backward()
        optimizer.step()

    return train_loss, train_emo_loss, train_domain_loss

@torch.no_grad()
def validate_and_test(model, data_loader, device, num_classes):
    model.eval()
    correct, total = 0, 0

    # unweighted accuracy
    unweightet_correct = [0] * num_classes
    unweightet_total = [0] * num_classes

    # weighted f1
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for batch in data_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["emo_labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)

        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)

        outputs, _ = model(feats, speech_padding_mask)

        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
        for i in range(len(labels)):
            unweightet_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                unweightet_correct[labels[i]] += 1
                tp[labels[i]] += 1
            else:
                fp[predicted[i]] += 1
                fn[labels[i]] += 1

    weighted_acc = correct / total * 100
    unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total) * 100
    weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total) * 100

    return weighted_acc, unweighted_acc, weighted_f1

def inference(model, ):
    pass


# def compute_unweighted_accuracy(list1, list2):
#     result = []
#     for i in range(len(list1)):
#         result.append(list1[i] / list2[i])
#     return sum(result)/len(result)

def compute_unweighted_accuracy(correct_list, total_list):
    result = []
    for c, t in zip(correct_list, total_list):
        if t == 0:
            continue     # ✅ 该类在 test 中不存在，跳过
        result.append(c / t)
    if len(result) == 0:
        return 0.0
    return sum(result) / len(result)

def compute_weighted_f1(tp, fp, fn, unweightet_total):
    f1_scores = []
    num_classes = len(tp)
    
    for i in range(num_classes):
        if tp[i] + fp[i] == 0:
            precision = 0
        else:
            precision = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall = 0
        else:
            recall = tp[i] / (tp[i] + fn[i])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
            
    wf1 = sum([f1_scores[i] * unweightet_total[i] for i in range(num_classes)]) / sum(unweightet_total)
    return wf1
