# -*- coding: utf-8 -*-
r"""
按【实时推理脚本】的特征定义，从采集日志生成 SVM 数据集（libsvm 稀疏格式）。

与课件 3. 特征提取和模型训练 里的 process_all_logs__24G_len228.py 的差别（重要）：
  - 训练脚本：38 个有效子载波、滑窗 80、特征 = 38 方差 + 3 统计量 = 41 维
  - 实时脚本：27 个有效子载波、滑窗 150、特征 = 27 方差 + 3 统计量 + 29 相邻子载波相关性 = 59 维
  课件自带的 CSI__50Hz27SBCW150.model 是 59 维的，即实时脚本那一套。
  本脚本直接复用课件的 DataPreprocess 类，保证训练/推理特征完全一致。

用法:
  python build_dataset_rt.py --out data\wall.txt --glob "穿墙场景\*.txt"
  python build_dataset_rt.py --bench 3000        # 只跑 3000 帧测速
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

RT_DIR = os.environ.get('CSI_RT_DIR', r'D:\Wifi CSI\wifi感知实践操作\4. 实战效果验证\SVM模型运行脚本')
sys.path.insert(0, RT_DIR)
from csi_data_preprocess_class_RTrun import DataPreprocess  # noqa: E402

# 与 RT_CSI_parseSVM_display.py 第 42 行完全一致
SUBC = [1, 3, 7, 9, 13, 15, 19, 21, 25, 27, 31, 33, 37, 39, 43, 45,
        49, 51, 55, 57, 61, 65, 67, 71, 73, 75, 77]
WIN = 150                      # DEFAULT_SLIDING_WINDOW_LEN
DIV = 10.0                     # normalize_csi_amplitude 的 div_factor
DATA_ROOT = r'D:\Wifi CSI\数据集\发出给高校学生'


def label_of(name):
    if 'exist' in name:
        return 1
    if 'none' in name:
        return -1
    raise ValueError('文件名不含 exist/none 标签: ' + name)


def parse_line(line):
    """CSI_DATA,seq,mac,rssi,rate,noise,agc,fft_gain,channel,ts,antenna,rx_format,len,first_invalid,"[v0,v1,...]"
    字段共 242 个：前 14 个是元信息，第 14 项起是 228 个 int8（带 "[ 前缀）
    返回 27 个有效子载波的模长（已 /10），或 None"""
    if not line.startswith('CSI_DATA'):
        return None
    parts = line.split(',')
    if len(parts) < 15:
        return None
    try:
        rx_format = int(parts[11])
        data_len = int(parts[12])
    except (ValueError, IndexError):
        return None
    if rx_format != 2 or data_len != 228:
        return None
    raw = ','.join(parts[14:]).strip().strip('"')
    if raw.startswith('['):
        raw = raw[1:]
    if raw.endswith(']'):
        raw = raw[:-1]
    try:
        vals = [int(x) for x in raw.split(',')]
    except ValueError:
        return None
    if len(vals) != data_len:
        return None
    ampl = []
    for i in range(0, data_len, 2):
        # 与课件一致：complex(虚部位置, 实部位置)，取模后与交换无关
        ampl.append(abs(complex(vals[i], vals[i + 1])) / DIV)
    return [ampl[i] for i in SUBC]


def run(files, out_path, bench=0, quiet=False):
    t0 = time.time()
    n_frames = n_feat = 0
    n_skip = 0
    per_file = []
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as fd:
        for path in files:
            name = os.path.basename(path)
            label = label_of(name)
            dp = DataPreprocess(len(SUBC), WIN, WIN)
            f_frames = f_feat = 0
            t_f = time.time()
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    mod = parse_line(line)
                    if mod is None:
                        n_skip += 1
                        continue
                    f_frames += 1
                    feats = dp.data_preprocess(mod)
                    if feats:
                        f_feat += 1
                        fd.write('%d %s\n' % (
                            label,
                            ' '.join('%d:%.6g' % (i + 1, v) for i, v in enumerate(feats))))
                    if bench and (n_frames + f_frames) >= bench:
                        break
            n_frames += f_frames
            n_feat += f_feat
            per_file.append((name, f_frames, f_feat))
            if not quiet:
                print('[%s] %s  帧=%d 样本=%d  %.1fs' %
                      ('exist' if label > 0 else 'none', name[:56], f_frames, f_feat,
                       time.time() - t_f), flush=True)
            if bench and n_frames >= bench:
                break
    el = time.time() - t0
    return dict(frames=n_frames, feats=n_feat, skip=n_skip, seconds=el,
                rate=n_frames / el if el else 0, per_file=per_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\Wifi CSI\实验工作区\data\dataset.txt')
    ap.add_argument('--glob', nargs='+', default=[r'穿墙场景\*.txt'])
    ap.add_argument('--bench', type=int, default=0, help='只处理前 N 帧用于测速')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    files = []
    for g in args.glob:
        files.extend(sorted(glob.glob(os.path.join(DATA_ROOT, g))))
    files = [f for f in files if f.lower().endswith('.txt')]
    if not files:
        print('没有匹配到文件')
        return 1
    print('待处理 %d 个文件' % len(files), flush=True)
    r = run(files, args.out, bench=args.bench, quiet=args.quiet)
    print('\n===== 汇总 =====')
    print('帧数 %d  样本 %d  跳过 %d  用时 %.1fs  速度 %.0f 帧/s'
          % (r['frames'], r['feats'], r['skip'], r['seconds'], r['rate']))
    print('输出: %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
