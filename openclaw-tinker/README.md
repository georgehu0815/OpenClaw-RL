# OpenClaw Tinker

OpenClaw 在 [Tinker](https://tinker.build) 云基础设施上的统一训练框架。通过单一入口点支持三种训练方法：

| 方法 | 参数 | 描述 |
|--------|------|-------------|
| **RL** | `--method rl` | 带过程奖励模型（PRM）评分和至少一次保证的 GRPO |
| **OPD** | `--method opd` | 基于事后提示和教师 log-prob 的在策略蒸馏 |
| **Combined** | `--method combine` | OPD 与 RL 优势的加权组合 |

## 快速开始

```bash
export TINKER_API_KEY="your-tinker-api-key"

# Combined 方法
python run.py --method combine --model-name Qwen/Qwen3-8B --prm-m 1 --batch-size 16 --w-opd 1.0 --w-rl 1.0

# RL 方法
python run.py --method rl --model-name Qwen/Qwen3-8B --prm-m 3 --batch-size 16

# OPD 方法
python run.py --method opd --model-name Qwen/Qwen3-8B --prm-m 1 --batch-size 16
```

## 架构

```
run.py                 CLI 入口点 (--method {rl, opd, combine})
├── config.py          统一的 TinkerConfig 数据类
├── trainer.py         训练循环：rollout → score → build datums → forward_backward → optim_step
│   ├── rollout.py     RolloutWorker：启动 API 代理，输入提示，收集补全
│   │   └── api_server.py   兼容 OpenAI 的代理，包含方法特定子类
│   ├── scorers.py     PRMScorer / OPDScorer / CombinedScorer
│   └── data_formatter.py   TrainingSample → Tinker Datum 转换
```

### 核心组件

- **Trainer** (`trainer.py`)：协调完整循环。创建两个 Tinker 客户端——一个 LoRA 训练客户端（策略模型）和一个基础采样客户端（教师/评判模型）。以可配置的间隔处理检查点保存和优雅关闭。

- **RolloutWorker** (`rollout.py`)：启动一个本地兼容 OpenAI 的 API 服务器，将请求转发给 Tinker 策略模型。外部环境（OpenClaw 任务）连接到该服务器。完成的会话进入评分和训练队列。

- **API Server** (`api_server.py`)：基类 `_BaseServer` 提供共享基础设施（Tinker 转发、鉴权、流式传输、分词、记录管理）。三个子类处理方法特定逻辑：
  - `OpenClawRLServer` — 带至少一次保证的 PRM 评分
  - `OpenClawOPDServer` — 提示评判 + 教师 log-prob，丢弃无 next_state 的轮次
  - `OpenClawCombineServer` — 三路分发（opd+rl / 仅 opd / 仅 rl）

- **Scorers** (`scorers.py`)：每个评分器评估已完成的会话，生成包含奖励和可选教师 log-prob 的 `TrainingSample` 对象。

- **Data Formatter** (`data_formatter.py`)：将 `TrainingSample` 批次转换为用于训练的 Tinker `Datum` 对象。RL/OPD 使用标量 GRPO 优势；Combined 按 token 计算 `w_opd * teacher_adv + w_rl * reward`。

## 配置

所有参数均可通过 CLI 参数或环境变量设置：

### 模型
| 参数 | 环境变量 | 默认值 | 描述 |
|------|---------|---------|-------------|
| `--model-name` | `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | 策略模型（必须是 Tinker 支持的） |
| `--lora-rank` | `LORA_RANK` | `32` | 训练的 LoRA 秩 |
| `--teacher-model-name` | `TEACHER_MODEL_NAME` | 与策略模型相同 | 教师/评判模型（基础版，无 LoRA） |

### 训练
| 参数 | 环境变量 | 默认值 | 描述 |
|------|---------|---------|-------------|
| `--learning-rate` | `LEARNING_RATE` | `1e-4` | 优化器学习率 |
| `--batch-size` | `BATCH_SIZE` | `4` | 每训练步的样本数 |
| `--max-steps` | `MAX_STEPS` | `1000` | 总训练步数 |
| `--loss-fn` | `LOSS_FN` | `ppo` | Tinker 损失函数：`ppo`、`importance_sampling`、`cispo` |
| `--kl-loss-coef` | `KL_LOSS_COEF` | `0.0` | KL 惩罚系数 |
| `--save-interval` | `SAVE_INTERVAL` | `20` | 每 N 步保存一次检查点 |
| `--resume-from-ckpt` | `RESUME_FROM_CKPT` | | 从检查点路径恢复训练 |

### 方法特定参数
| 参数 | 环境变量 | 默认值 | 方法 | 描述 |
|------|---------|---------|--------|-------------|
| `--w-opd` | `OPENCLAW_COMBINE_W_OPD` | `1.0` | combine | OPD 优势权重 |
| `--w-rl` | `OPENCLAW_COMBINE_W_RL` | `1.0` | combine | RL 优势权重 |
| `--eval-mode` | `EVAL_MODE` | `false` | opd | 在 OPD 旁启用 PRM 评估评分 |

### PRM / 提示评判
| 参数 | 环境变量 | 默认值 | 描述 |
|------|---------|---------|-------------|
| `--prm-m` | `PRM_M` | `3` | 评判采样数（多数投票） |
| `--prm-temperature` | `PRM_TEMPERATURE` | `0.6` | 评判采样温度 |
| `--prm-max-tokens` | `PRM_MAX_TOKENS` | `4096` | 评判响应最大 token 数 |

### 代理服务器
| 参数 | 环境变量 | 默认值 | 描述 |
|------|---------|---------|-------------|
| `--proxy-host` | `PROXY_HOST` | `0.0.0.0` | API 服务器绑定主机 |
| `--proxy-port` | `PROXY_PORT` | `30000` | API 服务器绑定端口 |
| `--served-model-name` | `SERVED_MODEL_NAME` | `qwen3-4b` | OpenAI API 响应中的模型名称 |
| `--api-key` | `SGLANG_API_KEY` | | 代理鉴权 API 密钥 |

## 训练方法

### RL（`--method rl`）

带过程奖励模型评分的标准 GRPO 强化学习：

1. 策略模型通过 API 代理生成响应
2. PRM 通过对下一状态评分来评估每一轮（对 M 个样本进行多数投票）
3. 奖励：`+1`（正确）、`-1`（错误）、`0`（不确定）
4. **至少一次保证**：若会话中所有轮次得分 ≤ 0，则最佳轮次获得奖励 = +1
5. GRPO 优势（标量奖励广播）→ Tinker Datum → 训练步骤

### OPD（`--method opd`）

利用事后提示和教师知识进行在策略蒸馏：

1. 策略模型生成响应；环境提供 next_state 观测
2. 提示评判器从 next_state 提取关键信息形成简洁提示
3. 教师模型（带提示上下文）对响应评分，获取 token 级 log-prob
4. 优势 = 来自教师的反向 KL：`-kl_coef * (student_lp - teacher_lp)`
5. 所有样本奖励 = 1.0（无显式奖励信号）
6. 可选 `--eval-mode`：同时计算 PRM 评估分数用于监控

### Combined（`--method combine`）

带三路样本分发的加权组合：

- **OPD+RL 样本**（同时有 next_state 和 reward）：获得两种优势分量
- **仅 OPD 样本**（有 next_state 但无 reward）：仅教师 KL 优势
- **仅 RL 样本**（有 reward 但无 next_state）：仅标量奖励优势

每 token 的组合优势：
```
combined_adv_i = w_opd * (-kl_coef * (student_lp_i - teacher_lp_i)) + w_rl * reward
```

## Tinker 集成

本项目使用 [Tinker](https://tinker.build) 云平台实现：

- **LoRA 训练**：`create_lora_training_client_async(base_model=..., rank=...)` 用于策略模型
- **采样**：`create_sampling_client_async(base_model=...)` 用于教师/评判模型
- **训练操作**：每步执行 `forward_backward_async()` + `optim_step_async()`
- **检查点**：`save_weights_and_get_sampling_client_async()` 更新策略采样客户端
- **损失函数**：支持 `ppo`、`importance_sampling`、`cispo`
