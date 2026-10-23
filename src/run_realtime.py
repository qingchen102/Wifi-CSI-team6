# -*- coding: utf-8 -*-
"""启动课件实时推理界面，并自动绕开 libsvm 不支持中文路径的问题。

libsvm 的 C 实现用 fopen 打开模型文件，遇到中文路径会静默失败（既不报错也读不到）。
本脚本先把模型复制到纯英文目录 D:\\wifi_csi_work\\model\\，再把该路径传给课件脚本。

用法:
  python run_realtime.py                     # 默认模型 + 自动找串口
  python run_realtime.py -p COM5
  python run_realtime.py -m "D:\\Wifi CSI\\实验工作区\\data\\model\\CSI_presence_v1.model"
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

COURSE_DIR = r'D:\Wifi CSI\wifi感知实践操作\4. 实战效果验证\SVM模型运行脚本'
ASCII_DIR = r'D:\wifi_csi_work\model'
DEFAULT_MODEL = r'D:\Wifi CSI\实验工作区\data\model\CSI_presence_v1.model'


def pick_port():
    try:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()
                 if 'CP210' in (p.description or '') or 'Silicon' in (p.description or '')]
        if ports:
            return ports[0]
    except Exception:
        pass
    return 'COM5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--port', default=None)
    ap.add_argument('-m', '--model', default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print('模型不存在: %s' % args.model)
        return 2

    os.makedirs(ASCII_DIR, exist_ok=True)
    ascii_model = os.path.join(ASCII_DIR, os.path.basename(args.model))
    shutil.copyfile(args.model, ascii_model)
    print('模型已就位: %s' % ascii_model)

    port = args.port or pick_port()
    cmd = [sys.executable, os.path.join(COURSE_DIR, 'RT_CSI_parseSVM_display.py'),
           '-p', port, '--model114', ascii_model, '--model228', ascii_model]
    print('启动: %s' % ' '.join(cmd))
    print('串口: %s  （若不对，用 -p 指定，例如 -p COM5）' % port)
    return subprocess.call(cmd, cwd=COURSE_DIR)


if __name__ == '__main__':
    sys.exit(main())
