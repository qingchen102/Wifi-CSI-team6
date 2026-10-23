# -*- coding: utf-8 -*-
"""按课件标准做法做随机 8:2 划分（按样本、分层），用于与"按文件分组划分"对照。

注意：滑窗步长为 1，相邻窗口重叠 149/150，随机划分会把几乎相同的窗口同时放进
训练集和测试集，因此准确率会明显偏高。这个数字对应课件的验收口径，
严格口径请看 train_eval.py 的分组划分结果。
"""
import glob
import os
import random

FEAT_DIR = r'D:\Wifi CSI\实验工作区\data\feat'
OUT_DIR = r'D:\Wifi CSI\实验工作区\data\random'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pos, neg = [], []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, '*.feat'))):
        if os.path.getsize(p) == 0:
            continue
        lab = 1 if 'exist' in os.path.basename(p) else -1
        with open(p, 'r', encoding='utf-8') as f:
            for ln in f:
                (pos if lab > 0 else neg).append(ln)

    rnd = random.Random(2026)
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    n_pos_tr = int(len(pos) * 0.8)
    n_neg_tr = int(len(neg) * 0.8)
    pos_tr_cap = int(80000 * len(pos) / float(len(pos) + len(neg)))
    neg_tr_cap = 80000 - pos_tr_cap
    train = pos[:pos_tr_cap] + neg[:neg_tr_cap]
    test = pos[n_pos_tr:][:20000] + neg[n_neg_tr:][:6000]
    rnd.shuffle(train)
    rnd.shuffle(test)
    with open(os.path.join(OUT_DIR, 'train.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(train)
    with open(os.path.join(OUT_DIR, 'test.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(test)
    print('正样本 %d（训练 %d / 测试 %d）' % (len(pos), n_pos_tr, len(pos) - n_pos_tr))
    print('负样本 %d（训练 %d / 测试 %d）' % (len(neg), n_neg_tr, len(neg) - n_neg_tr))
    print('合计训练 %d / 测试 %d' % (len(train), len(test)))


if __name__ == '__main__':
    main()
