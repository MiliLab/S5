import time
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Normal
# from sklearn.cluster import KMeans


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageVector(object):
    """Computes and stores the average and current value"""
    def __init__(self, dim=1):
        self.reset(dim)

    def reset(self, dim):
        self.avg = torch.ones(dim).cuda()
        self.sum = torch.zeros(dim).cuda() + 1e-5
        self.wgt = torch.zeros(dim).cuda() + 1e-5

    def update(self, index, value, wgt):
        self.sum[index] += value
        self.wgt[index] += wgt
        self.avg = self.sum / self.wgt


# # update class-wise memory banks
# def update_cls_memory(cls_memory, pred, label, memory_n_batches=128):
#     N, num_classes = pred.size()   # N x K

#     for i in range(num_classes):
#         if (label==i).sum() > 0:
#             n_samples = int((label==i).cpu().sum().numpy())
#             random_rows = np.random.choice(n_samples, size=max(n_samples//16, 1), replace=False)
#             pred_i = pred[label==i][random_rows]
#             if cls_memory[i].size(0) < memory_n_batches:
#                 cls_memory[i] = torch.cat((cls_memory[i], pred_i), dim=0)
#             else:
#                 cls_memory[i] = cls_memory[i][pred_i.size(0):]
#                 cls_memory[i] = torch.cat((cls_memory[i], pred_i), dim=0)

#     return cls_memory



# update class-wise memory banks
def update_cls_memory(cls_memory, pred, label, memory_n_batches=128):
    N, num_classes = pred.size()   # N x K

    for i in range(num_classes):
        if (label==i).sum() > 0:
            # n_samples = int((label==i).cpu().sum().numpy())
            # random_rows = np.random.choice(n_samples, size=max(n_samples//16, 1), replace=False)
            # pred_i = pred[label==i][random_rows]
            pred_i = pred[label==i]
            if len(cls_memory[i]) < memory_n_batches:
                cls_memory[i].append(pred_i)
            else:
                cls_memory[i].pop(0)
                cls_memory[i].append(pred_i)

    return cls_memory

# calculate class-wize distribution, including mean and covariance
def sample_cls_bins(cls_memory, softmax=False):
    num_classes, num_bins = len(cls_memory), 20
    cls_bins = torch.ones(num_classes, num_bins).cuda()

    for i in range(num_classes):
        if len(cls_memory[i]) != 0: 
            cls_memory_i = torch.cat(cls_memory[i], dim=0)
            cls_memory_i = cls_memory_i[:, i]

            sorted_memory_i, _ = torch.sort(cls_memory_i)
            sampled_indices = torch.linspace(0, len(sorted_memory_i)-1, num_bins+1).round().long()
            sorted_memory_i = sorted_memory_i[sampled_indices[1:]]

            if softmax:
                cls_bins[i] = F.softmax(sorted_memory_i, dim=-1)
            else:
                cls_bins[i] = sorted_memory_i

    return cls_bins

# # return dim: N
# def calc_wgt_bins(cls_bins, probs, labels, iters, total_iters):
#     num_bins = cls_bins.size(1)
#     cls_exts = cls_bins[labels]            # N x B
#     probs = probs.unsqueeze(dim=-1)        # N x 1
#     wgts = ((probs > cls_exts).sum(dim=-1)).clip(0, num_bins) / num_bins  # N

#     return wgts

def calc_wgt_bins(cls_bins, probs, labels, iters, total_iters):
    def custom_function(x, k):
        return 1 / (1 + torch.exp(-k * x))*2 - 1
        
    num_bins = cls_bins.size(1)
    cls_exts = cls_bins[labels]            # N x B
    probs = probs.unsqueeze(dim=-1)        # N x 1
    wgts = ((probs > cls_exts).sum(dim=-1)).clip(0, num_bins) / num_bins  # N

    k_parameter = iters / total_iters * 20 + 2
    # k_parameter = 3
    wgts = custom_function(wgts, k_parameter)

    return wgts


# # calculate class-wize distribution, including mean and covariance
# def calc_cls_centers(cls_memory, softmax=False):
#     num_classes = len(cls_memory)
#     cls_centers = torch.ones(num_classes, num_classes).cuda()

#     for i in range(num_classes):
#         if len(cls_memory[i]) != 0: 
#             cls_memory_i = torch.cat(cls_memory[i], dim=0)
#             if softmax:
#                 cls_centers[i] = F.softmax(cls_memory_i, dim=-1).mean(dim=0)
#             else:
#                 cls_centers[i] = cls_memory_i.mean(dim=0)
#     return cls_centers

def dataset_centers(cls_memory):
    num_classes = len(cls_memory)

    data_memory = []
    for i in range(num_classes):
        if len(cls_memory[i]) != 0: 
            cls_memory_i = torch.cat(cls_memory[i], dim=0)
            data_memory.append(cls_memory_i)
    data_memory = torch.cat(data_memory, dim=0)
    data_centers = data_memory.mean(dim=0)

    return data_centers


# # KL-based pseudo labels
# def calc_distributed_pseudo_labels(pred, cls_distr, thr=0):
#     N, num_classes = pred.size()

#     # kl pseudo labels
#     P = F.softmax(pred, dim=-1).detach().unsqueeze(dim=-2).cuda()      # N x K x K 
#     Q = F.softmax(cls_distr, dim=-1).detach().unsqueeze(dim=0).cuda()  # N x K x K 
#     M = (P + Q) / 2
#     kl = ((P*torch.log(P/M)).sum(dim=-1) + (Q*torch.log(Q/M)).sum(dim=-1)) / 2        # N x K
#     min_ent, pseudo_label_kl = kl.min(dim=-1)                                         # N, N

#     # casual pseudo labels
#     prob_debias = F.softmax(pred-cls_distr[pseudo_label_kl], dim=-1)
#     max_prob_debias, pseudo_label_debias = prob_debias.max(dim=-1)   # B x H x W

#     mask_match  = (pseudo_label_kl==pseudo_label_debias).float() 
#     mask_debais = (max_prob_debias > thr).float() 
#     mask        = mask_match * mask_debais

#     return pseudo_label_debias, mask

# cosine-based pseudo labels
def calc_distributed_pseudo_labels(pred, cls_center, cls_center_u, thr_low=0):
    N, num_classes = pred.size()

    # cosine pseudo labels
    cosine_map = F.cosine_similarity(pred.detach().unsqueeze(dim=-2), \
                                        cls_center.unsqueeze(dim=0), dim=-1)     # N x K
    max_cos, pl_cos = cosine_map.max(dim=-1)                                  # N
    mask_cos = max_cos.ge(0.75)
    prob_debias = pred - cls_center_u[pl_cos]
    max_prob_debias, pl_debias = prob_debias.max(dim=-1)   # N
    mask_debais = (max_prob_debias > thr_low) & (pl_debias==pl_cos)
    mask = (mask_cos * mask_debais).float()

    # casual pseudo labels
    # prob_debias = F.softmax(pred-cls_center_u[pseudo_label_cos], dim=-1)
    # max_prob_debias, pseudo_label_debias = prob_debias.max(dim=-1)   # N

    # mask_match  = (pseudo_label_cos==pseudo_label_debias).float() 
    # mask_debais = (max_prob_debias > thr).float() 
    # mask   = mask_cos

    return pl_cos, mask

def calc_clustered_pseudo_labels(pred, cls_center, cls_memory, thr=0):
    N, num_classes = pred.size()
    
    data_memory = []
    label_memory = []
    for k,v in cls_memory.items():
        cls_memory_i = torch.cat(v, dim=0)
        label_i = torch.zeros(cls_memory_i.size(0)).cuda() + k
        data_memory.append(cls_memory_i)
        label_memory.append(label_i)

    data_memory = torch.cat(data_memory, dim=0)
    label_memory = torch.cat(label_memory, dim=0)

    # cosine pseudo labels
    cosine_map = F.cosine_similarity(pred.detach().unsqueeze(dim=-2), \
                                      data_memory.unsqueeze(dim=0), dim=-1)     # N x N'
    pseudo_label_cos = cosine_map.max(dim=-1)[1]                                # N
    pseudo_label_cos = label_memory[pseudo_label_cos].long()                    # N

    # casual pseudo labels
    prob_debias = F.softmax(pred-cls_center[pseudo_label_cos], dim=-1)
    max_prob_debias, pseudo_label_debias = prob_debias.max(dim=-1)   # N

    mask_match  = (pseudo_label_cos==pseudo_label_debias).float() 
    mask_debais = (max_prob_debias > thr).float() 
    mask        = mask_match * mask_debais

    return pseudo_label_debias, mask

def kmeans_initilized_centers(cls_center, cls_memory):
    num_classes, _ = cls_center.size()
    cls_memory_slt = select_memory(cls_memory, 1000)

    refer_distribution = []
    for k, v in cls_memory_slt.items():
        refer_distribution.append(torch.cat(v, dim=0))
    refer_distribution = torch.cat(refer_distribution, dim=0).cpu().detach().numpy()

    # 自定义初始聚类中心
    initial_centers = cls_center.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=num_classes, init=initial_centers)
    kmeans.fit(refer_distribution)

    # 获取聚类结果
    new_label, new_cls_center = kmeans.labels_, kmeans.cluster_centers_
    new_cls_center = torch.tensor(new_cls_center).cuda()

    # 打印聚类结果
    # print("cls_centers: \n", cls_center)
    # print("new_cls_center: \n", new_cls_center)

    new_cls_memory = {}
    for i in range(num_classes):
        new_cls_memory[i] = [torch.tensor(refer_distribution[new_label==i]).cuda()]

    return new_cls_center, new_cls_memory


def calibrate_logits(cls_memory, logits, labels):
    num_classes = len(cls_memory)
    cls_mean, cls_std = torch.zeros(num_classes).cuda(), torch.ones(num_classes).cuda()

    for k, v in cls_memory.items():
        cls_memory_i = torch.cat(v, dim=0)
        cls_mean[k] = cls_memory_i.mean(dim=0)[k]
        cls_std[k] = cls_memory_i.std(dim=0)[k]
    new_mean, new_std = cls_mean.mean(), cls_std.mean()

    one_hot_labels = F.one_hot(labels, num_classes).float()
    re_logits = (logits - cls_mean[labels].unsqueeze(dim=-1))
    re_logits = re_logits + new_mean
    re_logits = re_logits*one_hot_labels + logits*(1-one_hot_labels)

    return re_logits