# -*- coding: utf-8 -*-
"""把各文件的 .feat 汇总成训练/测试集。

划分策略（防泄漏）：按【文件】分组划分，每个场景每个标签下每隔 5 个文件取 1 个进测试集（约 20%），
其余进训练集。这样同一段连续采集不会同时出现在训练和测试里，避免滑窗重叠导致的虚高准确率。

同时输出 manifest.csv 记录每个文件的场景、标签、样本数，便于分场景统计。
"""
import argparse
import csv
import glob
import math
import os
import random

FEAT_DIR = r'D:\Wifi CSI\实验工作区\data\feat'
OUT_DIR = r'D:\Wifi CSI\实验工作区\data'


def load_lines(path, cap):
    """均匀抽样读取，最多 cap 行（cap<=0 表示全读）"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f if ln.strip()]
    if cap and len(lines) > cap:
        step = len(lines) / float(cap)
        lines = [lines[min(int(i * step), len(lines) - 1)] for i in range(cap)]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-cap', type=int, default=4000, help='每个训练文件最多取多少样本')
    ap.add_argument('--test-cap', type=int, default=8000, help='每个测试文件最多取多少样本')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.feat')))
    if not files:
        print('没有 .feat 文件')
        return 1

    # 按 (场景, 标签) 分组
    groups = {}
    for p in files:
        base = os.path.basename(p)
        scene = base.split('__', 1)[0]
        label = 'exist' if 'exist' in base else ('none' if 'none' in base else '??')
        groups.setdefault((scene, label), []).append(p)

    manifest = []
    test_index = []
    off_te = 0
    tr = open(os.path.join(OUT_DIR, 'train.txt'), 'w', encoding='utf-8', newline='\n')
    te = open(os.path.join(OUT_DIR, 'test.txt'), 'w', encoding='utf-8', newline='\n')
    n_tr = n_te = 0
    for (scene, label), paths in sorted(groups.items()):
        paths.sort()
        for i, p in enumerate(paths):
            is_test = (i % 5 == 4)
            cap = args.test_cap if is_test else args.train_cap
            lines = load_lines(p, cap)
            fd = te if is_test else tr
            for ln in lines:
                fd.write(ln if ln.endswith('\n') else ln + '\n')
            n = len(lines)
            if is_test:
                n_te += n
                test_index.append(dict(file=os.path.basename(p), scene=scene, label=label,
                                       start=off_te, count=n))
                off_te += n
            else:
                n_tr += n
            manifest.append(dict(file=os.path.basename(p), scene=scene, label=label,
                                 split='test' if is_test else 'train', samples=n))
            print('%-5s %-8s %-6s %-56s %6d' % ('TEST' if is_test else 'train',
                                                scene, label, os.path.basename(p)[:56], n))
    tr.close()
    te.close()

    with open(os.path.join(OUT_DIR, 'manifest.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'scene', 'label', 'split', 'samples'])
        w.writeheader()
        w.writerows(manifest)

    with open(os.path.join(OUT_DIR, 'test_index.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'scene', 'label', 'start', 'count'])
        w.writeheader()
        w.writerows(test_index)

    print('\n训练样本 %d   测试样本 %d' % (n_tr, n_te))
    print('train.txt / test.txt / manifest.csv 已写入 %s' % OUT_DIR)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
