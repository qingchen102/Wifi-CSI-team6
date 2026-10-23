# -*- coding: utf-8 -*-
"""负样本误报专项评估：在全部 none 场景数据上模拟实时推理 + 25 窗多数表决，
统计误报窗口比例与"误报事件"次数（平滑输出由 无人 翻转为 有人 记一次事件）。

用法: python eval_false_alarm.py [--vote 25]
"""
import argparse
import glob
import os
import shutil
import sys
from collections import deque

from libsvm.svmutil import svm_load_model, svm_predict

FEAT_DIR = r'D:\Wifi CSI\实验工作区\data\feat'
MODEL = r'D:\Wifi CSI\实验工作区\data\model\CSI_presence_final.model'
ASCII = r'D:\wifi_csi_work\model\CSI_presence_final.model'


def load():
    os.makedirs(os.path.dirname(ASCII), exist_ok=True)
    shutil.copyfile(MODEL, ASCII)
    return svm_load_model(ASCII)


def rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            parts = ln.split()
            if not parts:
                continue
            y = int(float(parts[0]))
            x = {}
            for p in parts[1:]:
                i, v = p.split(':')
                x[int(i)] = float(v)
            yield y, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vote', type=int, default=25)
    args = ap.parse_args()

    model = load()
    files = [p for p in sorted(glob.glob(os.path.join(FEAT_DIR, '*none*.feat')))
             if os.path.getsize(p) > 0]
    tot_win = tot_fp_raw = tot_fp_smooth = tot_events = 0
    print('%-58s %7s %7s %7s %6s' % ('file', 'win', 'FP_raw', 'FP_sm', '事件'))
    for p in files:
        hist = deque(maxlen=max(1, args.vote))
        n = fp_raw = fp_smooth = events = 0
        prev = -1
        batch_y, batch_x = [], []
        for y, x in rows(p):
            batch_y.append(y)
            batch_x.append(x)
            if len(batch_x) >= 2000:
                _, _, vals = svm_predict(batch_y, batch_x, model, '-q')
                for yy, vv in zip(batch_y, vals):
                    d = vv[0]
                    raw = 1 if d > 0 else -1
                    n += 1
                    if raw > 0:
                        fp_raw += 1
                    hist.append(raw)
                    ratio = sum(1 for v in hist if v > 0) / float(len(hist))
                    smooth = 1 if ratio > 0.5 else -1
                    if smooth > 0:
                        fp_smooth += 1
                        if prev == -1:
                            events += 1
                    prev = smooth
                batch_y, batch_x = [], []
        if batch_x:
            _, _, vals = svm_predict(batch_y, batch_x, model, '-q')
            for yy, vv in zip(batch_y, vals):
                d = vv[0]
                raw = 1 if d > 0 else -1
                n += 1
                if raw > 0:
                    fp_raw += 1
                hist.append(raw)
                ratio = sum(1 for v in hist if v > 0) / float(len(hist))
                smooth = 1 if ratio > 0.5 else -1
                if smooth > 0:
                    fp_smooth += 1
                    if prev == -1:
                        events += 1
                prev = smooth
        print('%-58s %7d %7.3f%% %6.3f%% %6d'
              % (os.path.basename(p)[:58], n, 100.0 * fp_raw / n if n else 0,
                 100.0 * fp_smooth / n if n else 0, events))
        tot_win += n
        tot_fp_raw += fp_raw
        tot_fp_smooth += fp_smooth
        tot_events += events

    print('\n合计窗口 %d（约 %.2f 小时 @50Hz）' % (tot_win, tot_win / 50.0 / 3600))
    print('逐窗口误报率 %.4f%%（%d 窗）' % (100.0 * tot_fp_raw / tot_win, tot_fp_raw))
    print('表决后误报率 %.4f%%（%d 窗），误报事件 %d 次'
          % (100.0 * tot_fp_smooth / tot_win, tot_fp_smooth, tot_events))
    return 0


if __name__ == '__main__':
    sys.exit(main())
