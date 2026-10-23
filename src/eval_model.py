# -*- coding: utf-8 -*-
"""评估任意 libsvm 模型：与 train_eval.py 的测试集口径一致"""
import argparse, os, shutil, time, json
from libsvm.svmutil import svm_load_model, svm_predict

ASCII = r"D:\wifi_csi_work\model"

def load_feat(path, limit=None):
    y, x = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            y.append(float(parts[0]))
            d = {}
            for tok in parts[1:]:
                k, v = tok.split(":")
                d[int(k)] = float(v)
            x.append(d)
            if limit and len(y) >= limit:
                break
    return y, x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    os.makedirs(ASCII, exist_ok=True)
    tmp = os.path.join(ASCII, "_eval.model")
    shutil.copyfile(a.model, tmp)
    t0 = time.time()
    m = svm_load_model(tmp)
    t_load = time.time() - t0
    print("模型: %s (%.2f MB, %d SV, gamma=%.4f)" % (os.path.basename(a.model),
          os.path.getsize(a.model)/1048576, m.get_nr_sv(), m.param.gamma))

    y, x = load_feat(a.test, a.limit)
    print("测试样本 %d, 维度 %d" % (len(y), max(max(d) for d in x if d) if x else 0))
    t0 = time.time()
    p, acc, dec = svm_predict(y, x, m, "-q")
    acc = float(acc[0]) if isinstance(acc, (tuple, list)) else float(acc)
    dt = time.time() - t0

    tp = sum(1 for a_, b_ in zip(y, p) if a_ > 0 and b_ > 0)
    tn = sum(1 for a_, b_ in zip(y, p) if a_ < 0 and b_ < 0)
    fp = sum(1 for a_, b_ in zip(y, p) if a_ < 0 and b_ > 0)
    fn = sum(1 for a_, b_ in zip(y, p) if a_ > 0 and b_ < 0)
    n = len(y)
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    fpr = fp/(fp+tn) if fp+tn else 0
    tpr = rec
    tnr = tn/(tn+fp) if tn+fp else 0
    ba = (tpr+tnr)/2
    print("准确率 %.4f%%  (%d/%d)" % (acc, int(round(acc*n/100)), n))
    print("TP %d  TN %d  FP %d  FN %d" % (tp, tn, fp, fn))
    print("精确率 %.4f  召回率 %.4f  F1 %.4f  误报率 %.4f  平衡准确率 %.4f" % (prec, rec, f1, fpr, ba))
    print("加载 %.2fs  推理 %.1fs (%.0f 样本/秒)" % (t_load, dt, n/dt if dt else 0))
    out = dict(model=os.path.basename(a.model), test=os.path.basename(a.test), n=n,
               acc=acc, tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec,
               f1=f1, fpr=fpr, bal_acc=ba, n_sv=m.get_nr_sv(), gamma=m.param.gamma)
    with open(os.path.join(r"D:\Wifi CSI\实验工作区\data\model", "report_%s.json" % a.tag), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

