# -*- coding: utf-8 -*-
"""向量化 Hampel：把课件里 27 次逐子载波 Python 循环压成一次 numpy 计算。

课件原实现（hampel/extension/ext_hampel.py）对 41 点窗口做两重 Python 循环，
每帧要调用 27 次，单帧耗时 ~34 ms，占整个推理链路的 93%。
本模块只替换 Hampel 这一步，其余（滑窗/突变检测/带通/归一化/特征）完全复用课件代码，
因此输出与课件逐位可比。
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get('CSI_RT_DIR', r'D:\Wifi CSI\wifi感知实践操作\4. 实战效果验证\SVM模型运行脚本'))
from csi_data_preprocess_class_RTrun import DataPreprocess


class FastDataPreprocess(DataPreprocess):
    def __init__(self, csi_len, filter_wind_size=40, feat_seq_len=30):
        super().__init__(csi_len, filter_wind_size, feat_seq_len)
        self._buf = np.zeros((self.filter_wind_size + 1, self.csi_subcarrier_num), dtype=np.float64)
        self._cnt = 0

    def hampel_filter_serizl__lenAdapt(self, mod_list, filter_sigma=1.5):
        if len(mod_list) < 20:
            return []
        row = np.asarray(mod_list, dtype=np.float64)
        n = self.filter_wind_size + 1          # 41
        if self._cnt < n:
            self._buf[self._cnt, :] = row
            self._cnt += 1
            return []
        self._buf[:-1, :] = self._buf[1:, :]
        self._buf[-1, :] = row

        med = np.median(self._buf, axis=0)                    # 每个子载波的中位数
        mad = np.median(np.abs(self._buf - med), axis=0)      # 中位绝对偏差
        thr = filter_sigma * 1.4826 * mad
        center = self._buf[self.filter_wind_size // 2, :].copy()
        bad = np.abs(center - med) > thr
        center[bad] = med[bad]
        return center

