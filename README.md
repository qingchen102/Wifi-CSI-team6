# 基于 WiFi CSI 的人员存在感知系统

用一块 ESP32-C5 采集 WiFi 信道状态信息（CSI），判断房间里有没有人。完整链路：**CSI 采集 → 数据解析 → 预处理 → 特征提取 → SVM 训练 → 实时推理**。

不依赖摄像头、不依赖可穿戴设备，只利用人体对 WiFi 多径传播的扰动。

---

## 效果速览

| 指标 | 结果 |
|---|---|
| 存在检测准确率（全库随机 8:2） | **97.54%** |
| 分场景准确率 | 穿墙 **96.50%** / 公寓 **98.58%** / 小会议室 **99.54%** |
| 静态误报率（75 窗多数表决后） | **0.085%**（158,406 个无人窗口仅 2 次误触发） |
| 模型体积 | **7.06 MB**（9,447 个支持向量） |
| 端到端推理吞吐 | **1,015 帧/秒**（50 Hz 采样下余量约 20 倍） |
| 对比课件基线模型 | 基线 93.75%，本模型 97.54%（同一测试集） |

---

## 系统架构

```
路由器 AP（2.4 GHz / 20 MHz）
        │  ping 回包
        ▼
ESP32-C5 STA（CSI 采集固件）── USB 串口 921600 ──► CSI_DATA 原始日志（txt）
                                                        │
                        ┌───────────────────────────────┴───────────────────────────┐
                        ▼                                                           ▼
              离线训练支路                                                在线推理支路
   解析 → Hampel → 滑窗150 → 带通 → 归一化                        串口读流 → 同一预处理
        → 59 维特征 → SVM(RBF) 训练                                      → SVM 判决
        → 模型文件（7.06 MB）                                          → 多数表决 → 有人/无人
```

---

## 目录结构

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── build_dataset_rt.py      # 单文件特征提取：日志 → 59 维特征
│   ├── build_all.py             # 多进程批量提取
│   ├── fast_preprocess.py       # 向量化 Hampel（端到端提速 4.9 倍）
│   ├── make_split_rand.py       # 全库分层随机 8:2 划分
│   ├── make_split_scene.py      # 分场景独立划分
│   ├── make_split.py            # 按文件分组划分
│   ├── train_eval.py            # libsvm 训练 + 评估
│   ├── eval_model.py            # 评估任意模型（用于和基线对比）
│   ├── eval_false_alarm.py      # 静态误报专项评估
│   ├── bench_infer.py           # 推理耗时基准
│   ├── count_frames.py          # 统计日志帧数与时长
│   ├── rt_infer.py              # 实时/回放推理（串口）
│   └── run_realtime.py          # 一键启动课件图形界面
└── model/
    └── CSI_presence_final.model # 训练好的模型（C=10, γ=1, RBF）
```

---

## 环境准备

### 1. Python 依赖

```bash
python -m pip install -r requirements.txt
```

> ⚠️ 课件 `requirements.txt` 里的 `serial==0.0.97` 是空壳包，会和 `pyserial` 冲突，**不要装**。

### 2. 课件工具包路径

本仓库只包含我们自己写的代码。预处理核心类 `csi_data_preprocess_class_RTrun.py` 及其 `hampel/`、`slidingWindow.py` 属于课程资料，需要从课件目录取得。

默认路径写死在代码里，也可以直接用环境变量覆盖：

```powershell
# Windows PowerShell
$env:CSI_RT_DIR = "D:\Wifi CSI\wifi感知实践操作\4. 实战效果验证\SVM模型运行脚本"

# Linux / macOS
export CSI_RT_DIR="/path/to/SVM模型运行脚本"
```

---

## 快速开始

### 1. 采集数据

```bash
cd "wifi感知实践操作/2. CSI数据采集"
python Collect_CSI_DataSet.py -p COM5 --log "D:\my_data\教室"
```

Ctrl+C 停止，自动生成 `csi_log_月日_时分秒.txt`。

**采集要点**
- 路由器、开发板都固定在 1.2 m 高，视线无遮挡，相距 2–5 m
- 路由器需开启 **Wi-Fi5 兼容模式**（11ax 下 CSI 回调不触发）
- 开发板插 **USB-to-UART 口**（CP210x），上面原生 USB 口只输出固件日志
- 每段 10 分钟约 3 万帧、28 MB；开始后先静置 30 秒再计时

### 2. 生成数据集

```bash
python src/build_all.py --src "D:\my_data" --out ".\data\feat"
```

每个 `.txt` 生成一个 `.feat`，每行是 libsvm 稀疏格式：`label 1:val 2:val ... 59:val`。

> 文件名必须含 `exist`（有人）或 `none`（无人），否则标签判定会报错退出。

### 3. 划分训练/测试集

```bash
# 全库分层随机 8:2（课件口径）
python src/make_split_rand.py --feat-dir ".\data\feat" --out-dir ".\data\random"

# 分场景独立划分（更严格的泛化评估）
python src/make_split_scene.py --feat-dir ".\data\feat" --out-dir ".\data\per_scene"
```

### 4. 训练与评估

```bash
python src/train_eval.py \
  --train ".\data\random\train.txt" \
  --test  ".\data\random\test.txt" \
  --c 10 --g 1 --tag final
```

输出混淆矩阵、精确率/召回率/F1/误报率，并保存 `report_<tag>.json`。

### 5. 实时推理

```bash
# 串口实时
python src/rt_infer.py --port COM5 --model ".\model\CSI_presence_final.model" --vote 75

# 回放已有日志（验证链路）
python src/rt_infer.py --from-file ".\data\test_log.txt" --model ".\model\CSI_presence_final.model"

# 一键启动图形界面
python src/run_realtime.py --port COM5 --model ".\model\CSI_presence_final.model"
```

### 6. 误报专项评估

```bash
python src/eval_false_alarm.py --feat-dir ".\data\feat" --model ".\model\CSI_presence_final.model"
```

---

## 关键参数

| 参数 | 取值 | 说明 |
|---|---|---|
| 有效子载波数 | 27 | `[1,3,7,...,77]`，与课件实时脚本一致 |
| 滑动窗口 | 150 帧 | 50 Hz 下约 3 秒，覆盖一个呼吸周期 |
| Hampel 窗口 / n_sigma | 40 / 2.5 | 去除脉冲噪声 |
| 带通滤波 | 0.006 – 0.4 Hz | 保留呼吸与肢体活动频段 |
| 特征维度 | **59** | 27 方差 + 3 统计量 + 29 相邻子载波相关性 |
| 采样率 | 50 Hz | |
| 串口波特率 | 921600 | |
| SVM | C-SVC / RBF，C=10，γ=1 | |

---

## 与课件原版的差异

1. **统一特征口径**：课件训练脚本是 41 维（38 子载波、窗长 80），实时脚本是 59 维（27 子载波、窗长 150），两者不兼容。本项目统一采用**实时侧 59 维**，训练与推理共用同一份预处理代码，从根本上消除训练/部署偏移。

2. **向量化 Hampel**：课件 `hampel()` 是逐点 Python 双循环，每帧要算 4.5 万次 `np.median`，占整条链路 73%。`fast_preprocess.py` 只替换这一步（输出与课件逐位一致，最大偏差 9.1e-7），端到端从 **209 → 1,015 帧/秒**。

3. **多数表决抑制误报**：单窗口判决抖动大，取最近 75 个窗口（约 1.5 s）的多数票，把误报率从 0.954% 压到 0.085%。

4. **规避 libsvm 中文路径**：libsvm 的 C 实现打不开中文路径且静默失败，代码统一先复制到纯英文临时目录再加载。

---

## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 串口有启动日志但没有 `CSI_DATA` | 路由器是 Wi-Fi6 模式，改成 Wi-Fi5 兼容模式 |
| 完全看不到串口 | 插错 USB 口，改用下面的 CP210x 口（COM5） |
| `svm_load_model` 返回 None | 模型路径含中文，先复制到纯英文路径 |
| 训练脚本 `sys.exit(2)` | 文件名不含 `exist`/`none`，按标签重命名 |
| 样本数远少于帧数 | 正常，窗口内出现突变会被丢弃，保留率约 14% |
| 随机划分准确率虚高 | 滑窗步长为 1，相邻窗口高度重叠，应同时看分场景结果 |

---

## 数据说明

原始数据集约 2.15 GB（59 个日志、232 万帧、12.9 小时），未纳入版本库。模型文件（7 MB）已包含在 `model/` 下，可直接用于推理。
