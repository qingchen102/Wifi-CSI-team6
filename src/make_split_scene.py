# -*- coding: utf-8 -*-
"""分场景随机 8:2 划分，用于统计每个场景单独的准确率。"""
import glob
import os
import random

FEAT_DIR = r'D:\Wifi CSI\实验工作区\data\feat'
OUT_ROOT = r'D:\Wifi CSI\实验工作区\data\per_scene'
SCENES = ['穿墙场景', '公寓场景', '小会议室场景']


def main():
    for scene in SCENES:
        pos, neg = [], []
        for p in sorted(glob.glob(os.path.join(FEAT_DIR, scene + '__*.feat'))):
            if os.path.getsize(p) == 0:
                continue
            lab = 1 if 'exist' in os.path.basename(p) else -1
            with open(p, 'r', encoding='utf-8') as f:
                for ln in f:
                    (pos if lab > 0 else neg).append(ln)
        rnd = random.Random(2026)
        rnd.shuffle(pos)
        rnd.shuffle(neg)
        tr = pos[:int(len(pos) * 0.8)] + neg[:int(len(neg) * 0.8)]
        te = pos[int(len(pos) * 0.8):] + neg[int(len(neg) * 0.8):]
        rnd.shuffle(tr)
        rnd.shuffle(te)
        d = os.path.join(OUT_ROOT, scene)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'train.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(tr)
        with open(os.path.join(d, 'test.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(te)
        print('%s  正 %d / 负 %d  → 训练 %d 测试 %d' % (scene, len(pos), len(neg), len(tr), len(te)))


if __name__ == '__main__':
    main()
