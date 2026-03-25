# DeepSeek-V3 Pre-training on GB200 — Step-by-Step Guide

This guide reproduces **DeepSeek-V3 pre-training** on NVIDIA **GB200 NVL72** hardware using Megatron-LM with a stack of performance optimizations.

---

## Step 1 — Build the Docker Image

Start from `nvcr.io/nvidia/pytorch:25.09-py3` and install:

1. **System packages**: `git`, `tmux`, `libfabric-dev`, etc.
2. **Python packages**: `transformers`, `wandb`, `einops`, `sentencepiece`, test/lint tools
3. **cuDNN 9.14** — required for correct MXFP8 quantization and LayerNorm fusion (install `libcudnn9-cuda-13`)
4. **Custom Transformer Engine** — a patched fork at commit `7dd3914` (based on TE v2.9 + 2 extra PRs for quantization/CPU optimizations), built only for CUDA arch `100` (Blackwell)
5. **HybridEP (DeepEP)** — DeepSeek's expert parallelism library, checked out at commit `3f601f7`, built for arch `10.0`

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.09-py3 AS base

ENV SHELL=/bin/bash

RUN rm -rf /opt/megatron-lm && \
    apt-get update && \
    apt-get install -y sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev && \
    wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

RUN unset PIP_CONSTRAINT && pip install --no-cache-dir debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    setuptools==69.5.1

RUN apt-get update && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install libcudnn9-cuda-13

ARG COMMIT="7dd3914726abb79bc99ff5a5db1449458ed64151"
ARG TE="git+https://github.com/hxbai/TransformerEngine.git@${COMMIT}"
RUN pip install nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation --no-cache-dir $TE

WORKDIR /home/
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 3f601f7ac1c062c46502646ff04c535013bfca00 && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation .

RUN rm -rf /root/.cache /tmp/*
```

> **CUDA 12.9 alternative**: Change base to `nvcr.io/nvidia/pytorch:25.06-py3` and cuDNN to `libcudnn9-cuda-12`.

---

## Step 2 — Clone Megatron-LM

Use the `dev` branch after PR [#1917](https://github.com/NVIDIA/Megatron-LM/pull/1917):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout effebd81f410bc6566fffee6c320b6f8f762e06d
```

---

## Step 3 — Configure the Cluster

- You need **NVL72** nodes (72 GPUs per rack, connected via NVLink)
- EP=32 means every 32 GPUs (8 nodes) must be in the **same NVL domain/rack**
- With Slurm: add `--segment 8` to `sbatch` to enforce rack-local scheduling

```bash
# Example sbatch invocation
sbatch --segment 8 [... other sbatch args] your_training_script.sh
```

---

## Step 4 — Set Environment Variables

Set these before launching training:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVLINK_DOMAIN_SIZE=72
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=VERSION
export NCCL_GRAPH_REGISTER=0
```

| Variable | Purpose |
|---|---|
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Limits CUDA stream connections for overlap |
| `NVLINK_DOMAIN_SIZE=72` | Tells runtime the NVLink domain is 72 GPUs |
| `NCCL_NVLS_ENABLE=0` | Disables NVLink SHARP (not needed here) |
| `NVTE_FUSED_ATTN=1` | Enables fused attention in TE |
| `NVTE_NORM_FWD/BWD_USE_CUDNN=1` | Uses cuDNN for LayerNorm (requires cuDNN 9.14) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces memory fragmentation |

---

## Step 5 — Download `bindpcie`

```bash
wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie
chmod 755 bindpcie
```

This tool **binds each process to its nearest PCIe NIC**, critical for GB200 performance. Prefix every launch command with it — including inside Slurm `srun`:

```bash
#!/bin/bash
#SBATCH [... sbatch args]

srun [... srun args] /path/to/bindpcie /path/to/pretrain_gpt.py [... mcore arguments]
```

---

## Step 6 — Launch Training

Run `pretrain_gpt.py` with the full argument set. The model config matches DeepSeek-V3:

| Config | Value |
|---|---|
| Layers | 61 |
| Hidden size | 7168 |
| FFN hidden size | 18432 |
| Attention heads | 128 |
| Experts | 256 (MoE) |
| Active experts per token | top-8 |
| Sequence length | 4096 |
| Global batch size | 2048 |
| Total training samples | ~586M |
| Precision | MXFP8 (e4m3) |

Parallelism strategy: **TP1 × PP8 × EP32 × CP1** across the NVL72.

```bash
/path/to/bindpcie \
/path/to/megatron-lm/pretrain_gpt.py \
--distributed-timeout-minutes 60 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 8 \
--expert-model-parallel-size 32 \
--context-parallel-size 1 \
--expert-tensor-parallel-size 1 \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \
--use-mcore-models \
--sequence-parallel \
--use-flash-attn \
--disable-bias-linear \
--micro-batch-size 1 \
--global-batch-size 2048 \
--train-samples 585937500 \
--exit-duration-in-mins 220 \
--no-save-optim \
--no-check-for-nan-in-loss-and-grad \
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl te \
--manual-gc \
--manual-gc-interval 10 \
--enable-experimental \
--transformer-impl transformer_engine \
--seq-length 4096 \
--data-cache-path /path/to/data_cache \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model unsloth/DeepSeek-V3 \
--data-path /path/to/data \
--split 99,1,0 \
--no-mmap-bin-files \
--no-create-attention-mask-in-dataloader \
--num-workers 6 \
--num-layers 61 \
--hidden-size 7168 \
--ffn-hidden-size 18432 \
--num-attention-heads 128 \
--kv-channels 128 \
--max-position-embeddings 4096 \
--position-embedding-type rope \
--rotary-base 10000 \
--make-vocab-size-divisible-by 3232 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--swiglu \
--untie-embeddings-and-output-weights \
--multi-latent-attention \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--qk-layernorm \
--lr-decay-samples 584765624 \
--lr-warmup-samples 1536000 \
--lr-warmup-init 3.9e-7 \
--lr 3.9e-6 \
--min-lr 3.9e-7 \
--lr-decay-style cosine \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--num-experts 256 \
--moe-layer-freq ([0]*3+[1]*58) \
--moe-ffn-hidden-size 2048 \
--moe-shared-expert-intermediate-size 2048 \
--moe-router-load-balancing-type seq_aux_loss \
--moe-router-topk 8 \
--moe-grouped-gemm \
--moe-aux-loss-coeff 1e-4 \
--moe-router-group-topk 4 \
--moe-router-num-groups 8 \
--moe-router-pre-softmax \
--moe-router-padding-for-quantization \
--moe-router-topk-scaling-factor 2.5 \
--moe-router-score-function sigmoid \
--moe-router-enable-expert-bias \
--moe-router-bias-update-rate 1e-3 \
--moe-router-dtype fp32 \
--moe-permute-fusion \
--moe-router-fusion \
--q-lora-rank 1536 \
--kv-lora-rank 512 \
--qk-head-dim 128 \
--qk-pos-emb-head-dim 64 \
--v-head-dim 128 \
--rotary-scaling-factor 40 \
--mscale 1.0 \
--mscale-all-dim 1.0 \
--eval-iters 32 \
--eval-interval 200 \
--no-load-optim \
--no-load-rng \
--auto-detect-ckpt-format \
--load None \
--save /path/to/checkpoints \
--save-interval 500 \
--dist-ckpt-strictness log_all \
--init-method-std 0.02 \
--log-timers-to-tensorboard \
--log-memory-to-tensorboard \
--log-validation-ppl-to-tensorboard \
--log-throughput \
--log-interval 1 \
--logging-level 40 \
--tensorboard-dir /path/to/tensorboard \
--wandb-project deepseek-v3-benchmarking-v0.15 \
--wandb-exp-name DeepSeek-V3-TP1PP8EP32CP1VPP4-MBS1GBS2048-v0.15 \
--bf16 \
--enable-experimental \
--recompute-granularity selective \
--recompute-modules moe_act mlp \
--cuda-graph-impl transformer_engine \
--cuda-graph-scope attn moe_router moe_preprocess \
--te-rng-tracker \
--pipeline-model-parallel-layout "Et|(tt|)*30L" \
--moe-router-force-load-balancing \
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--moe-hybridep-num-sms 32 \
--fp8-recipe mxfp8 \
--fp8-format e4m3 \
--fp8-param-gather \
--reuse-grad-buf-for-mxfp8-param-ag \
--use-precision-aware-optimizer \
--main-grads-dtype fp32 \
--main-params-dtype fp32 \
--exp-avg-dtype bf16 \
--exp-avg-sq-dtype bf16
```

---

## Step 7 — Key Optimizations Explained

### Pipeline Layout
```bash
--pipeline-model-parallel-layout "Et|(tt|)*30L"
```
32 pipeline stages: `E` = Embedding, `t` = transformer layer, `L` = Loss.
- Stage 0: Embedding + 1 layer
- Stages 1–30: 2 layers each
- Stage 31: Loss

This uneven layout balances the overhead of embedding and loss layers across the pipeline.

### Selective Activation Recompute
```bash
--recompute-granularity selective
--recompute-modules moe_act mlp
```
Recomputes MoE activations and MLP during backward pass instead of storing them, trading compute for memory.

### Partial CUDA Graphs
```bash
--cuda-graph-impl transformer_engine
--cuda-graph-scope attn moe_router moe_preprocess
--te-rng-tracker
```
Captures attention, MoE router, and MoE preprocessing as CUDA graphs to reduce kernel launch overhead.

### HybridEP (Expert Parallelism)
```bash
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend hybridep
--moe-hybridep-num-sms 32
```
Uses DeepSeek's HybridEP library for efficient all-to-all token dispatch across expert-parallel ranks, allocating 32 SMs for communication.

### MXFP8 Training
```bash
--fp8-recipe mxfp8
--fp8-format e4m3
--fp8-param-gather
--reuse-grad-buf-for-mxfp8-param-ag
```
Uses microscaling FP8 (block-level quantization) for forward and backward passes. Gathers parameters in FP8 to reduce communication volume.

### Mixed-Precision Optimizer
```bash
--use-precision-aware-optimizer
--main-grads-dtype fp32
--main-params-dtype fp32
--exp-avg-dtype bf16
--exp-avg-sq-dtype bf16
```
Keeps master weights and gradients in FP32 for accuracy, but stores Adam moment estimates in BF16 to reduce optimizer memory footprint.

### Kernel Fusions
```bash
--cross-entropy-loss-fusion --cross-entropy-fusion-impl te
--moe-permute-fusion
--moe-router-fusion
```
Fuses cross-entropy, MoE token permutation, and router score computation into single kernels.

### Distributed Optimizer with Overlap
```bash
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```
Shards optimizer state across DP ranks and overlaps gradient reduction / parameter gather with compute.

### Manual Garbage Collection
```bash
--manual-gc
--manual-gc-interval 10
```
Disables Python's automatic GC and runs it every 10 steps manually, keeping ranks better synchronized and avoiding GC-induced bubbles.

---

## Summary

| Aspect | Choice |
|---|---|
| Hardware | GB200 NVL72 (72 GPUs/rack) |
| Parallelism | TP1 × PP8 × EP32 × CP1 |
| Precision | MXFP8 forward/backward, BF16 optimizer moments, FP32 master weights |
| Expert dispatch | HybridEP (DeepEP) |
| Memory saving | Selective recompute (MoE activations + MLP) |
| Throughput | CUDA graphs, kernel fusions, overlapped comms, PCIe binding |
