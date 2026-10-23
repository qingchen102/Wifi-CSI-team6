# -*- coding: utf-8 -*-
"""并行把三个场景的全部日志转成特征文件（每文件一个 .feat），供后续抽样合并训练。

输出目录: D:\Wifi CSI\实验工作区\data\feat\<场景>__<文件名>.feat
每行: <label> 1:v1 2:v2 ... 59:v59
"""
import glob
import os
import sys
import time
import traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_dataset_rt as B  # noqa: E402

OUTDIR = r'D:\Wifi CSI\实验工作区\data\feat'
SCENES = ['穿墙场景', '公寓场景', '小会议室场景']


def job(path):
    scene = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUTDIR, '%s__%s.feat' % (scene, stem))
    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, 'r', encoding='utf-8') as f:
            n = sum(1 for _ in f)
        return (scene, stem, -1, n, 0.0, 'skip')
    t0 = time.time()
    try:
        r = B.run([path], out, quiet=True)
        return (scene, stem, r['frames'], r['feats'], time.time() - t0, 'ok')
    except Exception:
        return (scene, stem, 0, 0, time.time() - t0, traceback.format_exc()[-300:])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = []
    for s in SCENES:
        files.extend(sorted(glob.glob(os.path.join(B.DATA_ROOT, s, '*.txt'))))
    print('文件数 %d' % len(files), flush=True)
    t0 = time.time()
    tot_f = tot_s = 0
    with Pool(processes=3) as pool:
        for i, (scene, stem, frames, feats, sec, status) in enumerate(
                pool.imap_unordered(job, files), 1):
            tot_f += max(frames, 0)
            tot_s += feats
            print('[%2d/%d] %-8s %-52s 帧=%-7d 样本=%-7d %5.1fs %s'
                  % (i, len(files), scene, stem[:52], frames, feats, sec,
                     '' if status == 'ok' else status), flush=True)
    el = time.time() - t0
    print('\n===== 完成 用时 %.1f min  帧=%d  样本=%d =====' % (el / 60, tot_f, tot_s))


if __name__ == '__main__':
    main()
