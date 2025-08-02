import argparse
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import utils
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, create_order_dataset
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import logging


def train(args, model, freeze_model, num_classes, device, train_label_loader, train_unlabel_loader, train_loader1, train_loader2, optimizer, m, epoch, tf_writer):
    model.train()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    unlabel_loader_iter = cycle(train_unlabel_loader)
    id_probs = AverageMeter('id_prob', ':.4e')
    ood_probs = AverageMeter('ood_prob', ':.4e')

    all_pesudo_every_class_num = torch.zeros(num_classes, dtype=torch.long).to(device)
    beta = torch.zeros(num_classes).to(device)
    num_batches = len(train_loader1)
    batch_size = args.batch_size
    sample_num = num_batches * batch_size
    feature_sum_list = torch.load("cifar100_labeled_mean_feature_tensor.pt")
    label_sample_max_sim_score = torch.load("cifar100_label_sample_max_sim_score.pt")
    all_label_sample_max_sim_score = []
    for i in label_sample_max_sim_score:
        all_label_sample_max_sim_score.extend(i)
    all_label_sample_max_sim_score = torch.tensor(all_label_sample_max_sim_score)
    mean = torch.mean(all_label_sample_max_sim_score)
    var = torch.var(all_label_sample_max_sim_score)

    m = min(m, 0.5)
    ce = MarginLoss(m=-1*m)

    for batch_idx, ((x1, x2, x3), target) in enumerate(train_label_loader):
        ((ux1, ux2, ux3), utarget) = next(unlabel_loader_iter)

        labeled_len = len(target)
        unlabeled_len = len(ux1)

        x1, x2, x3, target = x1.to(device), x2.to(device), x3.to(device), target.to(device)
        ux1, ux2, ux3, utarget = ux1.to(device), ux2.to(device), ux3.to(device), utarget.to(device)

        optimizer.zero_grad()
        
        # loss_supervised 
        logits1, feature1 = model(x1)
        prob1 = F.softmax(logits1, dim=1)
        max_prob1, pesudo_label = prob1.max(1)
        loss_supervised = ce(logits1, target)

        # loss_unsupervised
        m_beta = beta / (2-beta)
        threshold = m_beta * args.fixmatch_threshold

        x_weak = torch.cat((x2, ux2), dim=0)
        x_strong = torch.cat((x3, ux3), dim=0)
        logits_weak, feature_weak = model(x_weak)
        logits_strong, feature_strong = model(x_strong)
        prob_weak = F.softmax(logits_weak, dim=1)
        prob_strong = F.softmax(logits_strong, dim=1)
        max_prob_weak, pesudo_label_weak = prob_weak.max(1)
        max_prob_strong, pesudo_label_strong = prob_strong.max(1)
        mask = torch.where(max_prob_weak > threshold[pesudo_label_weak], torch.ones_like(max_prob_weak), torch.zeros_like(max_prob_weak))

        p = torch.sum(prob_weak[mask.bool()], dim=0)
        p = p / torch.sum(prob_weak[mask.bool()])
        q = torch.ones(num_classes)
        q = q / num_classes
        align = torch.log(p / q.to(device))

        loss_unsupervised = (F.cross_entropy(logits_strong + align, pesudo_label_weak, reduction='none') * mask).mean()

        pro_over_fixmatch_threshold = torch.where(max_prob_weak > args.fixmatch_threshold, pesudo_label_weak, -1*torch.ones_like(pesudo_label_weak))
        elements, counts = torch.unique(pro_over_fixmatch_threshold, return_counts=True)
        elements, counts = elements[1:], counts[1:]
        pesudo_every_class_num = torch.zeros(num_classes, dtype=torch.long).to(device)
        pesudo_every_class_num[elements] = counts
        all_pesudo_every_class_num = all_pesudo_every_class_num + pesudo_every_class_num
        max_all_pesudo_every_class_num, _ = all_pesudo_every_class_num.max(0)
        no_pesudo_label_sample_num = sample_num - torch.sum(all_pesudo_every_class_num)

        if max_all_pesudo_every_class_num > no_pesudo_label_sample_num:
            max_value = max_all_pesudo_every_class_num
        else:
            max_value = no_pesudo_label_sample_num
        beta = all_pesudo_every_class_num / max_value
        
        
        x1 = torch.cat([x1, ux1], 0)
        x2 = torch.cat([x2, ux2], 0)

        output, feat = model(x1)
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)
        
        # loss_pairsimilarity 
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)
        pos_pairs = []
        target_np = target.cpu().numpy()

        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)

        _, freeze_feat = freeze_model(x1)
        unlabel_feat = freeze_feat[labeled_len : ]
        pred_list = [1 for i in range(labeled_len)]
        for i in range(len(unlabel_feat)):
            unlabel_feat_i = torch.repeat_interleave(unlabel_feat[i], repeats=len(feature_sum_list)).reshape(len(feature_sum_list), 512)
            cos_sim = F.cosine_similarity(unlabel_feat_i, feature_sum_list.to(device))
            values, indices = cos_sim.max(0)
            if values < mean:
                pred_list.append(1) 
            else:
                pred_list.append(0) 
        pred_list = torch.tensor(pred_list).to(device)
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        loss_pairsim = bce(pos_sim, ones)
        loss_pairsim = (F.binary_cross_entropy(pos_sim, ones, reduction='none') * pred_list).mean()

        # loss_regularization
        loss_regularization = - entropy(torch.mean(prob, 0))
        loss = loss_supervised + loss_regularization + loss_unsupervised + loss_pairsim

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        _, preds = prob.max(1)
        targets = torch.cat((target, utarget), dim=0)
        preds = preds.cpu().numpy().astype(int)
        targets = targets.cpu().numpy().astype(int)
        overall_acc = cluster_acc(preds, targets)


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    probs = np.array([])

    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            probs = np.append(probs, prob.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    print(epoch)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)

    mean_uncert = 1 - np.mean(confs)

    return mean_uncert

def main():
    parser = argparse.ArgumentParser(description='RSSL_NACH')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str,
                        default='name')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio, download=True,
                                                    transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                      labeled_ratio=args.labeled_ratio, download=True,
                                                      transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                      unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                             labeled_ratio=args.labeled_ratio, download=True,
                                             transform=datasets.dict_transform['cifar_test'],
                                             unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 10

        train_data_set = train_label_set + train_unlabel_set

    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True,
                                                     transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                       transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                              labeled_ratio=args.labeled_ratio, download=True,
                                              transform=datasets.dict_transform['cifar_test'],
                                              unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100

        train_data_set = train_label_set + train_unlabel_set

    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    train_loader1 = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                drop_last=True)
    train_loader2 = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                drop_last=True)

    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)

    freeze_model = models.resnet18(num_classes=num_classes)
    freeze_model = freeze_model.to(device)

    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    freeze_model.load_state_dict(state_dict, strict=False)
    freeze_model = freeze_model.to(device)

    # whether lock the backbone
    for name, param in freeze_model.named_parameters():
        param.requires_grad = False

    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in range(args.epochs):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        train(args, model, freeze_model, num_classes, device, train_label_loader, train_unlabel_loader, train_loader1,
              train_loader2, optimizer, mean_uncert, epoch, tf_writer)
        scheduler.step()
        print("#######################################")
    tf_writer.close()

if __name__ == '__main__':
    main()







