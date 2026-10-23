# -*- coding: utf-8 -*-
"""训练 + 测试 SVM（libsvm），输出总准确率、分场景准确率、混淆矩阵、误报率。

用法:
  python train_eval.py --grid                 # 先网格搜索再训练
  python train_eval.py --c 32 --g 2           # 直接用课件参数
"""
import argparse
import csv
import json
import os
import shutil
import random
import sys
import time

from libsvm.svmutil import svm_train, svm_predict, svm_read_problem, svm_save_model

DATA = r'D:\Wifi CSI\实验工作区\data'
MODEL_DIR = r'D:\Wifi CSI\实验工作区\data\model'
ASCII_TMP = r'D:\wifi_csi_work\model'


def read_libsvm(path):
    return svm_read_problem(path)


def grid_search(y, x, cs, gs, folds, sub):
    idx = list(range(len(y)))
    random.Random(42).shuffle(idx)
    idx = idx[:sub]
    ys = [y[i] for i in idx]
    xs = [x[i] for i in idx]
    best = (None, None, -1)
    rows = []
    for g in gs:
        for c in cs:
            t0 = time.time()
            acc = svm_train(ys, xs, '-s 0 -t 2 -c %g -g %g -v %d -q' % (c, g, folds))
            rows.append((c, g, acc))
            print('  C=%-7g gamma=%-9g CV=%.4f%%  (%.1fs)' % (c, g, acc, time.time() - t0), flush=True)
            if acc > best[2]:
                best = (c, g, acc)
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=os.path.join(DATA, 'train.txt'))
    ap.add_argument('--test', default=os.path.join(DATA, 'test.txt'))
    ap.add_argument('--c', type=float, default=None)
    ap.add_argument('--g', type=float, default=None)
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--sub', type=int, default=12000, help='网格搜索用的样本数')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print('读取训练集 %s ...' % args.train, flush=True)
    y, x = read_libsvm(args.train)
    print('训练样本 %d，特征维度 %d' % (len(y), len(x[0])), flush=True)
    print('读取测试集 %s ...' % args.test, flush=True)
    yt, xt = read_libsvm(args.test)
    print('测试样本 %d' % len(yt), flush=True)

    grid_rows = []
    C, G = args.c, args.g
    if args.grid:
        print('\n===== 网格搜索（%d 折，%d 样本）=====' % (args.folds, min(args.sub, len(y))), flush=True)
        (bc, bg, bacc), grid_rows = grid_search(
            y, x, [1, 10, 100, 1000], [0.01, 0.1, 1, 10, 100, 1000, 10000], args.folds,
            min(args.sub, len(y)))
        print('最佳: C=%g gamma=%g CV=%.4f%%' % (bc, bg, bacc), flush=True)
        C, G = bc, bg
    if C is None:
        C = 32.0
    if G is None:
        G = 2.0

    print('\n===== 训练最终模型  C=%g gamma=%g =====' % (C, G), flush=True)
    t0 = time.time()
    param = '-s 0 -t 2 -c %g -g %g -q' % (C, G)
    model = svm_train(y, x, param)
    train_sec = time.time() - t0
    model_path = os.path.join(MODEL_DIR, 'CSI_presence_%s.model' % args.tag)
    # libsvm 的 C 实现无法写中文路径（静默失败），先写 ASCII 临时路径再拷回
    safe_tag = ''.join(c if (c.isascii() and (c.isalnum() or c in '_-')) else '_' for c in args.tag)
    tmp_model = os.path.join(ASCII_TMP, 'CSI_presence_%s.model' % safe_tag)
    os.makedirs(ASCII_TMP, exist_ok=True)
    svm_save_model(tmp_model, model)
    shutil.copyfile(tmp_model, model_path)
    print('训练耗时 %.1fs，模型 %s (%.2f MB)' %
          (train_sec, model_path, os.path.getsize(model_path) / 1024 / 1024), flush=True)

    print('\n===== 测试集预测 =====', flush=True)
    p_label, p_acc, p_val = svm_predict(yt, xt, model, '-q')
    acc, mse, scc = p_acc
    n_correct = sum(1 for a, b in zip(yt, p_label) if (a > 0) == (b > 0))
    print('总准确率 = %.4f%% (%d/%d)' % (acc, n_correct, len(yt)), flush=True)

    # 混淆矩阵
    tp = sum(1 for a, b in zip(yt, p_label) if a > 0 and b > 0)
    tn = sum(1 for a, b in zip(yt, p_label) if a < 0 and b < 0)
    fp = sum(1 for a, b in zip(yt, p_label) if a < 0 and b > 0)
    fn = sum(1 for a, b in zip(yt, p_label) if a > 0 and b < 0)
    print('TP=%d TN=%d FP=%d FN=%d' % (tp, tn, fp, fn), flush=True)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    print('精确率=%.4f 召回率=%.4f F1=%.4f 误报率(FPR)=%.4f' % (prec, rec, f1, fpr), flush=True)

    # 分场景 / 分文件
    per = []
    idx_path = os.path.join(DATA, 'test_index.csv')
    if os.path.exists(idx_path):
        with open(idx_path, 'r', encoding='utf-8-sig') as _f:
            _tot = sum(int(r['count']) for r in csv.DictReader(_f))
        if _tot != len(yt):
            idx_path = ''   # 索引与当前测试集不匹配（例如随机划分），跳过逐文件统计
    if idx_path and os.path.exists(idx_path):
        with open(idx_path, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                s, n = int(r['start']), int(r['count'])
                seg_y = yt[s:s + n]
                seg_p = p_label[s:s + n]
                ok = sum(1 for a, b in zip(seg_y, seg_p) if (a > 0) == (b > 0))
                per.append(dict(file=r['file'], scene=r['scene'], label=r['label'],
                                n=n, correct=ok, acc=ok / n if n else 0))
                print('  %-8s %-6s %-50s %6d  %.4f%%' %
                      (r['scene'], r['label'], r['file'][:50], n, 100.0 * ok / n), flush=True)

    # 汇总
    summary = dict(C=C, G=G, train_n=len(y), test_n=len(yt), dim=len(x[0]),
                   train_sec=train_sec, model_path=model_path,
                   model_size_mb=os.path.getsize(model_path) / 1024 / 1024,
                   acc=acc, n_correct=n_correct, tp=tp, tn=tn, fp=fp, fn=fn,
                   precision=prec, recall=rec, f1=f1, fpr=fpr,
                   n_sv=getattr(model, 'l', None), grid=grid_rows, per_file=per)
    out_json = os.path.join(MODEL_DIR, 'report_%s.json' % args.tag)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n报告已写入 %s' % out_json, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
