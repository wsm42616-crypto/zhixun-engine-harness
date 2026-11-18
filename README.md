# 智训引擎：SDK v2 试运行候选源码

本源码对应已通过本地 Linux 检查的 SDK 1.0.2 镜像。尚未在真实比赛宿主调用大模型或完成平台评测，不宣称具有正式成绩。本文用于创建可供评审访问的源码仓库；本 ZIP 不是 Docker 镜像，也不是学校报送压缩包。

## 构建和入口

```sh
docker build --platform linux/amd64 -t zhixun-harness:sdk-v2 .
docker run --rm --network none zhixun-harness:sdk-v2 describe
```

固定入口为 `python3 /app/harness.py describe|train|predict`。训练由真实宿主通过 stdin start 帧发起，不能以没有宿主的手工 train 命令冒充平台运行。

SDK 原包安装在镜像内；官方离线测试可在 `/opt/contest-sdk` 执行：

```sh
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
python3 -m unittest discover -s /app/platform_tests -v
```

模型访问仅为 SDK 的 host_gateway_rpc；Key 由用户填写在比赛平台的密钥栏，镜像不索取、不保存 Key。当前提交配置路线为基础模型 Qwen/Qwen3.8-27B，没有自有大模型 Sampler 微调权重。

## 任务与策略

A 是六类气体分类，主指标 Macro-F1 越高越好。B 是轴承剩余寿命回归（分钟），C 是发动机剩余寿命回归（循环），两者 RMSE 越低越好。任务与方向必须符合宿主冻结合同。

一进程处理平台当前一轮：验证父检查点、读历史、识别公开输入均值漂移和特征偏移线索、模型提出受限参数方案、保留 Adam/缩放状态真实续训、提交候选、接收测量、模型选择、保存状态、finish/ack。每轮两次网关模型请求，单次输出上限 min(4096, 宿主授权上限)；响应错误停止，不自动重试。无改善候选保留父模型；满足监测条件时可恢复更早的实际产物，在当前公开验证集重测后决定回滚，历史分数不直接当作当前分数。

宿主没有提供可比父/候选分数时，明确标注使用本地公开验证结果作开发选择，不伪造宿主分数。所有选择以真实模型回执、真实训练报告和平台可审计产物为依据。预测从 /model 加载指定 checkpoint.npz，逐行回复并刷新。

## 目录

- `harness.py`：平台原始 SDK 模板。
- `contest_submission.py`：本项目接入与受限实验/选择策略。
- `industrial/`：从平台 A/B/C 开发包提取且保持字节一致的 CPU 基线和适配器。
- `sdk/`：平台提供的 contest_harness 1.0.2，以及原始离线测试和协议示例测试夹具；不含 test_live.py。
- `platform_tests/`：合成数据下的协议与实际续训检查，不计为比赛实验。
- `provenance/`：来源和文件摘要；不包含数据或权重。

## 验证与复现边界

Linux 普通用户、只读根、可写 /workspace 和 /tmp、关闭外网条件下，官方 15 项离线 SDK 测试与本项目 8 项接入测试通过，A/B/C 原始验证指标和独立 SDK 预测复核一致。实际镜像依赖为 Python 3.12.16、NumPy 1.26.4 和 contest_harness 1.0.2。

Dockerfile 的基础标签可随时间变化。已经验证的镜像来源、上游平台清单摘要、配置摘要和镜像内容哈希记录在 provenance 中；外部重新构建不自动等同于此次已验证镜像。测试不能代替真实宿主、网关、云端实验规范和隐藏集评测。

## 来源与记录

平台 SDK 和原始工业基线来自赛题提供的文件，来源摘要见 provenance；Python 和 NumPy 为基础依赖。接入和策略开发使用 Codex 辅助编写、审阅与测试代码；比赛运行阶段拟使用的平台模型是 PyTRiO 的 Qwen3.8-27B，二者用途不同。

既有 A 题开发记录在 https://swanlab.cn/@flameboy/zhixun-engine ，三次真实模型指导的开发闭环并非本 SDK 版真实平台评测，也不是大模型本身的微调记录。SDK 版尚待真实宿主/网关验证，以及完整云端记录和正式规范核对。

源码不含 Key、真实训练集、权重、虚拟环境、完整运行环境或隐藏标签。不将本地复核 JSON 冒充官方验证 JSON。
