# -*- coding: utf-8 -*-
"""实时存在感知推理（串口版 / 回放版），带时域多数表决。

流程与课件 RT_CSI_parseSVM_display.py 完全一致（27 个有效子载波 → 模长/10 →
Hampel 去噪 → 滑窗 150 → 归一化 → 27 方差 + 3 统计量 + 29 相关性 = 59 维 → SVM），
额外增加了两件工程上必要的事：
  1) 模型先复制到纯英文目录再加载（libsvm 的 C 实现不支持中文路径，会静默失败）；
  2) 时域多数表决：单窗口判决抖动大，取最近 N 个窗口的多数票作为最终结论，
     可把孤立误报压到接近 0，满足"静态长时间无误报"。

用法:
  python rt_infer.py --port COM5                       # 实时
  python rt_infer.py --from-file 某个日志.txt           # 回放验证
  python rt_infer.py --from-file x.txt --vote 1         # 关闭表决（逐窗口输出）
"""
import argparse
import collections
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_dataset_rt import parse_line, SUBC, WIN  # noqa: E402
from csi_data_preprocess_class_RTrun import DataPreprocess  # noqa: E402
from libsvm.svmutil import svm_load_model, svm_predict  # noqa: E402

ASCII_DIR = r'D:\wifi_csi_work\model'
DEFAULT_MODEL = r'D:\Wifi CSI\实验工作区\data\model\CSI_presence_final.model'


def load_model(path):
    os.makedirs(ASCII_DIR, exist_ok=True)
    dst = os.path.join(ASCII_DIR, os.path.basename(path))
    shutil.copyfile(path, dst)
    m = svm_load_model(dst)
    if getattr(m, 'l', None) is None:
        raise RuntimeError('模型加载失败（路径或文件损坏）: %s' % dst)
    return m


class Predictor:
    def __init__(self, model, vote=75):
        self.model = model
        self.dp = DataPreprocess(len(SUBC), WIN, WIN)
        self.hist = collections.deque(maxlen=max(1, vote))
        self.n_frame = 0
        self.n_pred = 0

    def feed(self, line):
        """喂一行原始 CSI_DATA；返回 (raw_label, smooth_label, vote_ratio) 或 None"""
        mod = parse_line(line)
        if mod is None:
            return None
        self.n_frame += 1
        feats = self.dp.data_preprocess(mod)
        if not feats:
            return None
        self.n_pred += 1
        x = {i + 1: float(v) for i, v in enumerate(feats)}
        _, _, vals = svm_predict([1], [x], self.model, '-q')
        d = vals[0][0]
        raw = 1 if d > 0 else -1
        self.hist.append(raw)
        ratio = sum(1 for v in self.hist if v > 0) / float(len(self.hist))
        smooth = 1 if ratio > 0.5 else -1
        return raw, smooth, ratio, d


def run_file(path, model, vote, quiet=False):
    pr = Predictor(model, vote)
    t0 = time.time()
    last_print = 0
    stats = collections.Counter()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            r = pr.feed(line)
            if r is None:
                continue
            raw, smooth, ratio, d = r
            stats[smooth] += 1
            if not quiet and pr.n_pred - last_print >= 50:
                print('窗口 %6d  决策值 %+.3f  单窗=%s  平滑=%s (有人票 %.0f%%)'
                      % (pr.n_pred, d, '有人' if raw > 0 else '无人',
                         '有人' if smooth > 0 else '无人', 100 * ratio), flush=True)
    el = time.time() - t0
    total = stats[1] + stats[-1]
    print('\n回放完成：帧 %d，判决 %d，用时 %.1fs（%.0f 帧/s）'
          % (pr.n_frame, pr.n_pred, el, pr.n_frame / el if el else 0))
    if total:
        print('平滑后：有人 %d (%.1f%%)，无人 %d (%.1f%%)'
              % (stats[1], 100.0 * stats[1] / total, stats[-1], 100.0 * stats[-1] / total))


def run_serial(port, baud, model, vote):
    import serial
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.dtr = False
    ser.rts = False
    ser.timeout = 1
    ser.open()
    print('已打开 %s @ %d，等待 CSI 数据...' % (port, baud))
    pr = Predictor(model, vote)
    try:
        while True:
            line = ser.readline().decode('utf-8', 'ignore')
            if not line.startswith('CSI_DATA'):
                continue
            r = pr.feed(line)
            if r is None:
                continue
            raw, smooth, ratio, d = r
            if pr.n_pred % 25 == 0:
                print('%s 窗口%6d 决策值%+.3f 单窗=%s 平滑=%s (有人票 %.0f%%)'
                      % (time.strftime('%H:%M:%S'), pr.n_pred, d,
                         '有人' if raw > 0 else '无人',
                         '有人' if smooth > 0 else '无人', 100 * ratio), flush=True)
    except KeyboardInterrupt:
        print('\n已停止')
    finally:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--port', default='COM5')
    ap.add_argument('-b', '--baud', type=int, default=921600)
    ap.add_argument('-m', '--model', default=DEFAULT_MODEL)
    ap.add_argument('--from-file', default=None)
    ap.add_argument('--vote', type=int, default=75, help='多数表决窗口数，1 表示关闭')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    model = load_model(args.model)
    print('模型已加载: %s' % args.model)
    if args.from_file:
        run_file(args.from_file, model, args.vote, args.quiet)
    else:
        run_serial(args.port, args.baud, model, args.vote)
    return 0


if __name__ == '__main__':
    sys.exit(main())
