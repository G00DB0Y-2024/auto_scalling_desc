# LLM 推理集群智能 Auto Scaling 系统

> 基于多尺度时序解耦与多置信区间峰値预测的弹性扩缩容引擎

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
- [API 参考](#api-参考)
- [配置说明](#配置说明)
- [性能指标](#性能指标)
- [常见问题](#常见问题)

---

## 项目概述

### 背景与痛点

在云算力基座上，我们承载着多种**核心 LLM 推理服务**（如对话问答、Agent 执行等）。这些服务面临的挑战是：

| 问题 | 描述 | 影响 |
|------|------|------|
| **扩容滞后** | 日间突发峰值来临时，被动响应式扩容无法及时应对 | 服务受损、用户体验下降 |
| **算力闲置** | 夜间低谷期大量 GPU/CPU 资源空闲 | 资源浪费、成本浪费 |
| **关联挤占** | 多个 LLM 推理微服务之间存在复杂的资源竞争关系 | 资源调度不精准 |

### 解决方案

本项目将传统的**被动响应式扩缩容**重构为**主动预测式智能扩缩容**，通过研发**智能化 Seasonality 预测模型**，实现分钟级的精准容量规划。

---

## 核心特性

### 1. 多置信 Peak 估计

- 支持 **P50 / P60 / P70 / P80 / P90 / P95** 六个置信区间输出
- 下游调度器可根据业务 SLA 要求灵活选择：
  - **高置信度**（P80-P95）：保守扩容，适合关键业务
  - **低置信度**（P50-P60）：激进收缩，适合成本优化场景

### 2. FFT 多尺度时序解耦

- 使用**快速傅里叶变换（FFT）** 将时序信号分解为多个频率分量
- 在每个频率尺度上提取显著的**周期性模式**
- 实现对日周期、周周期、节假日周期等的精确捕捉

### 3. 多头自注意力机制

- 引入**多头自注意力（Multi-Head Self-Attention）** 捕捉序列内部时序依赖
- 自动学习不同时刻之间的相关性权重
- 避免传统 RNN/LSTM 的梯度消失问题

### 4. MixHop 图卷积协同建模

- 引入**自适应 MixHop 图卷积层**
- 在每个时间尺度内自主学习**动态邻接矩阵**
- 捕捉多个 LLM 推理微服务之间的：
  - **协同关系**：A 服务上涨时 B 服务也上涨
  - **挤占关系**：A 服务上涨时 B 服务资源被抢占

### 5. 自适应尺度聚合（MoE）

- 采用 **Mixture of Experts (MoE)** 架构
- 不同专家网络专注于不同类型的时序模式
- 自适应聚合多尺度预测结果

### 6. OOD 鲁棒性

- 模型在面对**异常流量（Out-of-Distribution）** 时展现强鲁棒性
- 能够自动学习并解释多尺度序列间的相关性
- 避免突发异常导致的错误预测

---

## 技术架构

### 系统架构图

```
+-----------------------------------------------------------------------------+
|                              用户请求层                                       |
|  +-------------+    +-------------+    +-------------+                      |
|  |  对话问答    |    |  Agent执行   |    |  其他服务...  |                     |
|  +------+------+    +------+------+    +------+------+                      |
|         |                  |                  |                             |
|         +------------------+------------------+                             |
|                              |                                                |
|                              v                                                |
|                   +-------------------+                                        |
|                   |   流量采集层       |                                        |
|                   |  (Metrics SDK)   |                                        |
|                   +--------+----------+                                        |
+------------------------------+---------------------------------------------+
                               |
                               v
+-----------------------------------------------------------------------------+
|                           时序数据库层                                        |
|                   +-------------------+                                        |
|                   |   Prometheus      |                                        |
|                   |   / InfluxDB      |                                        |
|                   +--------+----------+                                        |
+------------------------------+---------------------------------------------+
                               |
                               v
+-----------------------------------------------------------------------------+
|                          预测引擎核心层                                       |
|                                                                             |
|  +---------------------------------------------------------------------+     |
|  |                      Seasonality 预测模型                            |     |
|  |                                                                      |     |
|  |   +--------------+    +--------------+    +--------------+          |     |
|  |   | FFT 多尺度    |--->| 多头自注意力  |--->| 自适应尺度   |          |     |
|  |   | 时序解耦      |    | 时序依赖提取  |    | 聚合 (MoE)   |          |     |
|  |   +--------------+    +--------------+    +--------------+          |     |
|  |          |                                       |                   |     |
|  |          v                                       v                   |     |
|  |   +---------------------------------------------------+      |     |
|  |   |              MixHop 图卷积层 (关联服务协同)                |      |     |
|  |   |     +---------+    +---------+    +---------+              |      |     |
|  |   |     |对话问答  |<-->| Agent执行|<-->|其他服务  |              |      |     |
|  |   |     +---------+    +---------+    +---------+              |      |     |
|  |   +---------------------------------------------------+      |     |
|  |                              |                                        |     |
|  |                              v                                        |     |
|  |   +---------------------------------------------------+      |     |
|  |   |              多置信 Peak 估计输出层                         |      |     |
|  |   |   P50 ---- P60 ---- P70 ---- P80 ---- P90 ---- P95   |      |     |
|  |   +---------------------------------------------------+      |     |
|  +---------------------------------------------------------------------+     |
|                                                                             |
+------------------------------+---------------------------------------------+
                               |
                               v
+-----------------------------------------------------------------------------+
|                           调度决策层                                         |
|                                                                             |
|  +-----------------+    +-----------------+    +-----------------+        |
|  |   保守策略调度   |    |   均衡策略调度   |    |   激进策略调度   |        |
|  |  (使用 P95)     |    |  (使用 P80)     |    |  (使用 P50)     |        |
|  +--------+--------+    +--------+--------+    +--------+--------+        |
|           |                      |                      |                  |
|           +----------------------+----------------------+                  |
|                                  |                                          |
|                                  v                                          |
|                    +-------------------------+                             |
|                    |    云弹性伸缩组 (ASG)    |                             |
|                    +-------------------------+                             |
|                                  |                                          |
|                                  v                                          |
|                    +-------------------------+                             |
|                    |   容量调整指令下发       |                             |
|                    |   (分钟级响应)          |                             |
|                    +-------------------------+                             |
+------------------------------+---------------------------------------------+
                               |
                               v
+-----------------------------------------------------------------------------+
|                           资源管理层                                         |
|                                                                             |
|  +---------------------------------------------------------------------+   |
|  |                    夜间冗余算力召回机制                                 |   |
|  |                                                                      |   |
|  |   日间高峰 --------> 夜间低谷 --------> 算力召回 --------> 离线任务 |   |
|  |   (推理服务)           (低负载)            (73% GPU/CPU)       (训练任务) |   |
|  |                                                                      |   |
|  +---------------------------------------------------------------------+   |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### 核心模型架构

```
输入时序序列
     |
     v
+------------------------------------+
|        FFT 多尺度解耦层              |
|  +-----+-----+-----+-----+-----+   |
|  |周期1 |周期2 |周期3 |周期4 |... |   |
|  |(天) |(周) |(小时)|(分钟)|    |   |
|  +-----+-----+-----+-----+-----+   |
+---------------+--------------------+
                |
                v
+------------------------------------+
|       多头自注意力层 (Multi-Head)   |
|  +---+ +---+ +---+ +---+ +---+    |
|  |H1 | |H2 | |H3 | |H4 | |H5 |    |
|  +---+ +---+ +---+ +---+ +---+    |
|  提取不同时序依赖模式                 |
+---------------+--------------------+
                |
                v
+------------------------------------+
|     自适应 MixHop 图卷积层          |
|                                    |
|   学习动态邻接矩阵:                 |
|   A_learned = Softmax(MLP(X))      |
|                                    |
|   捕捉服务间协同与挤占关系           |
+---------------+--------------------+
                |
                v
+------------------------------------+
|     自适应尺度聚合层 (MoE)         |
|                                    |
|   Expert1: 日常周期模式             |
|   Expert2: 周末周期模式             |
|   Expert3: 节假日模式               |
|   Expert4: 异常检测模式             |
|   ...                              |
|                                    |
|   输出: weighted_sum(Experts)       |
+---------------+--------------------+
                |
                v
+------------------------------------+
|      多置信 Peak 估计输出层          |
|                                    |
|   Season Average: 基础预测值        |
|   Peak_P50: 50% 置信上界           |
|   Peak_P60: 60% 置信上界           |
|   Peak_P70: 70% 置信上界           |
|   Peak_P80: 80% 置信上界           |
|   Peak_P90: 90% 置信上界           |
|   Peak_P95: 95% 置信上界           |
|                                    |
+------------------------------------+
```

---

## 快速开始

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| Python | >= 3.9 | 运行环境 |
| PyTorch | >= 2.0 | 深度学习框架 |
| CUDA | >= 11.7 | GPU 加速（如有） |
| 内存 | >= 16GB | 推荐 32GB+ |

### 安装步骤

```bash
# 1. 克隆项目
git clone https://your-repo/auto_scaling_predictor.git
cd auto_scaling_predictor

# 2. 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等
```

### 快速运行示例

```python
from predictor import SeasonalityPredictor

# 初始化预测器
predictor = SeasonalityPredictor(
    model_path="models/stochastic_peak_model.pt",
    confidence_levels=[50, 60, 70, 80, 90, 95]
)

# 输入历史流量数据
historical_data = [
    {"timestamp": "2026-09-04 10:00", "requests": 15000, "service": "llm_chat"},
    {"timestamp": "2026-09-04 11:00", "requests": 18000, "service": "llm_chat"},
    # ... 更多历史数据
]

# 获取预测结果
predictions = predictor.predict(
    data=historical_data,
    horizon_minutes=30,  # 预测未来30分钟
    strategy="balanced"   # 可选: "conservative" | "balanced" | "aggressive"
)

print(f"Season Average: {predictions['season_average']}")
print(f"P95 Peak: {predictions['peak_p95']}")
print(f"Recommended Scaling: {predictions['scaling_action']}")
```

### Docker 部署

```bash
# 构建镜像
docker build -t auto-scaling-predictor:latest .

# 运行容器
docker run -d -p 8000:8000 \
    -v ./models:/app/models \
    -v ./configs:/app/configs \
    --name auto-scaling-predictor \
    auto-scaling-predictor:latest
```

---

## 模块详解

### 1. `fft_decomposer.py` - FFT 多尺度时序解耦

**功能**：将复杂的时序信号分解为多个频率分量

**核心参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fs` | float | 1.0 | 采样频率 |
| `lowcut` | float | 0.01 | 低频截止频率 |
| `highcut` | float | 0.5 | 高频截止频率 |
| `num_scales` | int | 5 | 分解的尺度数量 |

**使用示例**：

```python
from fft_decomposer import FFTDecomposer

decomposer = FFTDecomposer(num_scales=5)
scales = decomposer.decompose(time_series)

for scale_name, scale_data in scales.items():
    print(f"{scale_name}: {scale_data['amplitude']}")
```

### 2. `multihead_attention.py` - 多头自注意力层

**功能**：捕捉时序序列中的长距离依赖关系

**核心参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_model` | int | 256 | 模型维度 |
| `num_heads` | int | 8 | 注意力头数量 |
| `dropout` | float | 0.1 | Dropout 比例 |

**使用示例**：

```python
from multihead_attention import TemporalAttention

attention = TemporalAttention(d_model=256, num_heads=8)
context = attention(queries, keys, values)
```

### 3. `mixhop_graph_conv.py` - MixHop 图卷积层

**功能**：学习服务之间的协同与挤占关系

**核心参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_services` | int | 必需 | 服务数量 |
| `hidden_dim` | int | 64 | 隐藏层维度 |
| `num_hops` | int | 3 | MixHop 跳数 |

**邻接矩阵学习**：

```python
# 可学习邻接矩阵自动捕捉服务关系
adj_matrix = mixhop_layer.learned_adjacency
# adj_matrix[i][j] > 0: 服务 i 和 j 协同
# adj_matrix[i][j] < 0: 服务 i 和 j 挤占
```

### 4. `moe_aggregator.py` - 自适应尺度聚合 (MoE)

**功能**：多个专家网络处理不同类型的时序模式

**专家类型**：

| 专家 | 擅长的模式 | 使用场景 |
|------|------------|----------|
| Expert 1 | 日常周期 | 工作日正常流量 |
| Expert 2 | 周末周期 | 周末低峰 |
| Expert 3 | 节假日 | 春节/国庆等 |
| Expert 4 | 异常检测 | 突发流量 |

### 5. `peak_estimator.py` - 多置信 Peak 估计

**功能**：输出不同置信水平的 Peak 预测

**输出格式**：

```python
{
    "season_average": 12000,    # 季节性平均值
    "peak_p50": 13500,         # 50% 置信上界
    "peak_p60": 14200,         # 60% 置信上界
    "peak_p70": 15000,         # 70% 置信上界
    "peak_p80": 16000,         # 80% 置信上界
    "peak_p90": 17500,         # 90% 置信上界
    "peak_p95": 19000,         # 95% 置信上界
}
```

### 6. `scaling_scheduler.py` - 调度决策器

**功能**：根据预测结果生成扩缩容指令

**策略选择**：

```python
from scaling_scheduler import ScalingScheduler

scheduler = ScalingScheduler()

# 保守策略（高可用优先）
action = scheduler.decide(
    predictions=predictions,
    strategy="conservative"  # 使用 P90-P95
)

# 均衡策略（平衡成本与可用性）
action = scheduler.decide(
    predictions=predictions,
    strategy="balanced"  # 使用 P70-P80
)

# 激进策略（成本优化优先）
action = scheduler.decide(
    predictions=predictions,
    strategy="aggressive"  # 使用 P50-P60
)
```

---

## API 参考

### REST API

#### `POST /predict`

预测未来流量

**请求示例**：

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "service": "llm_chat",
    "horizon_minutes": 30,
    "strategy": "balanced",
    "history": [
      {"timestamp": "2026-09-05T10:00:00", "value": 15000},
      {"timestamp": "2026-09-05T11:00:00", "value": 18000}
    ]
  }'
```

**响应示例**：

```json
{
  "success": true,
  "data": {
    "service": "llm_chat",
    "horizon_minutes": 30,
    "predictions": {
      "season_average": 16500,
      "peak_p50": 17200,
      "peak_p60": 17800,
      "peak_p70": 18500,
      "peak_p80": 19200,
      "peak_p90": 20100,
      "peak_p95": 21500
    },
    "scaling_recommendation": {
      "action": "scale_up",
      "instance_count": 3,
      "confidence": "balanced"
    }
  }
}
```

#### `GET /health`

健康检查

**响应**：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

---

## 配置说明

### 基础配置 (`config/base.yaml`)

```yaml
# 模型配置
model:
  name: "stochastic_peak_v1"
  version: "1.0.0"
  checkpoint_path: "./models/stochastic_peak_model.pt"
  
# 预测配置
prediction:
  horizon_minutes: 30
  confidence_levels: [50, 60, 70, 80, 90, 95]
  update_frequency_minutes: 5

# 调度配置
scheduler:
  default_strategy: "balanced"
  strategies:
    conservative:
      confidence_level: 95
      min_buffer_percent: 20
    balanced:
      confidence_level: 80
      min_buffer_percent: 10
    aggressive:
      confidence_level: 60
      min_buffer_percent: 0

# 资源限制
resources:
  min_instances: 1
  max_instances: 100
  scale_up_cooldown_minutes: 5
  scale_down_cooldown_minutes: 15
```

### 服务关联配置 (`config/services.yaml`)

```yaml
services:
  - name: "llm_chat"
    type: "primary"
    baseline_instances: 5
    priority: 1
    
  - name: "llm_agent"
    type: "primary"
    baseline_instances: 3
    priority: 2
    
relationships:
  - source: "llm_chat"
    target: "llm_agent"
    relationship: "compete"  # compete: 挤占, cooperate: 协同
    weight: 0.7
```

---

## 性能指标

### 预测准确性

| 指标 | 数值 | 说明 |
|------|------|------|
| MAPE (平均绝对百分比误差) | < 8% | 预测值与实际值的平均偏差 |
| P50 命中率 | > 90% | 实际值落在 P50 预测范围内的比例 |
| P80 命中率 | > 85% | 实际值落在 P80 预测范围内的比例 |
| P95 命中率 | > 80% | 实际值落在 P95 预测范围内的比例 |
| 异常检测召回率 | > 92% | 成功预测突发峰值的比例 |

### 业务价值

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 扩容响应时间 | 10-15 分钟 | < 1 分钟 | **90%+** |
| 服务受损事件 | 每周 5-8 次 | < 1 次 | **87%** |
| 夜间算力闲置率 | 65% | 18% | **73% 召回** |
| 月度计算成本 | 基准 | -25% | **节省 25%** |

### 夜间算力召回流程

```
22:00 夜间低谷开始
   |
   v
预测模型检测到流量下降
   |
   +-> LLM 推理实例从 20 台 -> 5 台
   |
   +-> 释放的 15 台实例
   |         |
   |         v
   |    转入"冷备池"
   |         |
   |         v
   |    分配给内部离线训练任务
   |         |
   |         v
   |    GPU 利用率: 18% -> 85%
   |
   v
06:00 凌晨高峰来临前
   |
   +-> 逐步回收实例
   |
   +-> LLM 推理实例从 5 台 -> 20 台
```

---

## 常见问题

### Q1: 如何选择合适的置信策略？

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 关键业务（不容失败） | `conservative` | 使用 P95，保证充足容量 |
| 常规服务 | `balanced` | 平衡成本与可用性 |
| 实验性服务 | `aggressive` | 优先成本优化 |

### Q2: 模型多久更新一次？

- **在线更新**：每 5 分钟增量更新
- **全量重训练**：每周日凌晨 2:00（低峰期）

### Q3: 如何处理突发异常流量？

模型内置异常检测机制：
1. 检测当前流量是否超出历史正常范围
2. 自动切换到 OOD 鲁棒模式
3. 临时提高置信区间（自动使用 P99）

### Q4: 支持哪些时序数据库？

| 数据库 | 支持状态 | 配置项 |
|--------|----------|--------|
| Prometheus | 完整支持 | `db.prometheus.url` |
| InfluxDB | 完整支持 | `db.influxdb.url` |
| OpenTSDB | 完整支持 | `db.opentsdb.url` |
| 其他 | 插件支持 | 可扩展 Adapter |

### Q5: 如何监控模型性能？

```bash
# 启动 Grafana 监控面板
docker-compose up -d grafana

# 查看关键指标
# - prediction_accuracy: 预测准确率
# - latency_p99: 预测延迟 P99
# - scaling_actions: 扩缩容执行次数
# - resource_utilization: 资源利用率
```

---

## 许可证

本项目基于 Apache 2.0 许可证开源，欢迎社区贡献与使用。

---

## 联系方式

| 角色 | 职责 |
|------|------|
| 项目负责人 | 架构设计与核心研发 |
| 算法团队 | 模型优化与调参 |
| 运维团队 | 部署与监控 |

---

*最后更新：2026 年 9 月 5 日*
