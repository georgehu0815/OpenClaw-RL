---
title: "simple\\_rl v2 — System Architecture"
subtitle: "LoRA Training · Adapter Continuity · W\\&B Observability"
date: "2026-03-29"
author: "OpenClaw-RL Project"
geometry: "margin=1.5cm"
fontsize: 11pt
mainfont: "Helvetica Neue"
colorlinks: true
header-includes:
  - \usepackage{xcolor}
  - \usepackage{fancyhdr}
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{array}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\textcolor{gray}{\small simple\_rl v2 Architecture}}
  - \fancyhead[R]{\textcolor{gray}{\small \thepage}}
  - \fancyfoot[C]{\textcolor{gray}{\small OpenClaw-RL · 2026}}
  - \renewcommand{\headrulewidth}{0.3pt}
  - \definecolor{cBlue}{HTML}{4B8BDE}
  - \definecolor{cOrange}{HTML}{E07B39}
  - \definecolor{cPurple}{HTML}{9B72DE}
  - \definecolor{cGreen}{HTML}{4BAD6E}
  - \definecolor{cDBlue}{HTML}{4B72DE}
  - \definecolor{cViolet}{HTML}{BB72DE}
  - \definecolor{cGold}{HTML}{D4A017}
  - \definecolor{cRed}{HTML}{DE4B4B}
  - \definecolor{cLavender}{HTML}{7B52DE}
  - \definecolor{cYellow}{HTML}{D4D44B}
  - \newcommand{\swatch}[1]{\colorbox{#1}{\phantom{XX}}}
---

\begin{center}
\includegraphics[width=\textwidth,keepaspectratio]{simple-rl-architecture-v2.png}
\end{center}

\newpage

## Component Legend

\begin{tabular}{lll}
\toprule
\textbf{Swatch} & \textbf{Component} & \textbf{Role} \\
\midrule
\swatch{cBlue}     & \texttt{train\_async.py}         & AsyncIO orchestrator, adapter continuity state \\
\swatch{cOrange}   & \texttt{LocalEnvPool / TerminalEnv} & Docker container pool \\
\swatch{cPurple}   & \texttt{agent\_loop.py}          & Multi-turn agent, semaphore, retry \\
\swatch{cGreen}    & oMLX Policy Server               & LLM serving on port 8080 \\
\swatch{cDBlue}    & \texttt{rollout\_buffer.py}      & GRPO advantage computation \\
\swatch{cViolet}   & \texttt{prm\_client.py}          & Optional per-step scoring + JSONL \\
\swatch{cGold}     & \texttt{mlx\_grpo\_bridge.py}    & Subprocess orchestrator \textbf{(NEW)} \\
\swatch{cRed}      & \texttt{grpo\_worker.py}         & mlx-tune training subprocess \textbf{(NEW)} \\
\swatch{cLavender} & Weights \& Biases                & Observability sink \\
\swatch{cYellow}   & Dataset                          & Task definitions \\
\bottomrule
\end{tabular}

\vspace{1em}

## What Is New in v2 (marked `<-- NEW in v2` in diagram)

| Node | Enhancement |
|------|-------------|
| Policy Semaphore | `asyncio.Semaphore(1)` serialises all oMLX API calls |
| Exponential Backoff Retry | 3x retry on HTTP 500 with 2 s / 4 s / 8 s delays |
| `prm_steps.jsonl` | Per-turn PRM votes + representative evaluation text |
| `mlx_grpo_bridge.py` | Subprocess bridge; serialises job JSON, reads result JSON |
| `grpo_worker.py` | Full mlx-tune training subprocess under dedicated venv |
| `load_adapter(prev_path)` | Resumes LoRA weights from previous round's checkpoint |
| LoRA Verification | Trainable-param count + rank/alpha logged to stdout and W\&B |
| `wandb.define_metric` | Sets `grpo/step` as x-axis so loss charts render correctly |
| `_last_adapter_path` | Module-level state that chains adapter checkpoints across rounds |

## Round-to-Round Adapter Continuity Flow

```
Round 1:  from_pretrained(base) -> get_peft_model() [no prev adapter]
            -> train -> save -> logs/grpo_adapters/round_0001/

Round 2:  from_pretrained(base) -> get_peft_model()
            -> load_adapter(round_0001/) [fill lora_a, lora_b]
            -> train -> save -> logs/grpo_adapters/round_0002/

Round N:  from_pretrained(base) -> get_peft_model()
            -> load_adapter(round_000N-1/) [accumulate all prior updates]
            -> train -> save -> logs/grpo_adapters/round_000N/
```

Each round's saved adapter is the **cumulative sum** of all gradient updates
applied across rounds 1 through N.  The base model weights are never modified.
