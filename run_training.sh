./export-weights-bias.sh
POLICY_URL=http://localhost:8080/v1 PRM_ENABLE=1 bash simple_rl/run.sh
# --dry-run for a quick test (2 rounds, no real LLM)
# export WANDB_KEY="your-wandb-key"
# ./run_training.sh 
# === simple-rl startup ===
# [✓] Docker daemon reachable
# [✓] Policy server reachable at http://localhost:8080/v1

# Configuration:
#   POLICY_URL       = http://localhost:8080/v1
#   POLICY_MODEL     = Qwen3-8B-4bit
#   MAX_CONCURRENT   = 4
#   N_SAMPLES        = 8
#   ROLLOUT_BATCH    = 4
#   MAX_TURNS        = 20
#   DATASET          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/data/sample_tasks.jsonl
#   LOG_DIR          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/logs


# wandb: [wandb.login()] Using explicit session credentials for https://api.wandb.ai.
# wandb: No netrc file found, creating one.
# wandb: Appending key for api.wandb.ai to your netrc file: /Users/ghu/.netrc
# wandb: Currently logged in as: bochuxt7 (bochuxt7-iot) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
# wandb: Tracking run with wandb version 0.25.1
# wandb: Run data is saved locally in /Volumes/ExternalSSD/train/OpenClaw-RL/wandb/run-20260329_154334-lv7f7cuj
# wandb: Run `wandb offline` to turn off syncing.
# wandb: Syncing run pleasant-wildflower-1
# wandb: ⭐️ View project at https://wandb.ai/bochuxt7-iot/terminal-rl-simple
# wandb: 🚀 View run at https://wandb.ai/bochuxt7-iot/terminal-rl-simple/runs/lv7f7cuj

# https://wandb.ai/bochuxt7-iot/terminal-rl-simple/runs/lv7f7cuj?nw=nwuserbochuxt7