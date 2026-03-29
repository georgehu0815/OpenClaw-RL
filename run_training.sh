POLICY_URL=http://localhost:8080/v1 bash simple_rl/run.sh
# --dry-run for a quick test (2 rounds, no real LLM)

./run_training.sh 
=== simple-rl startup ===
[✓] Docker daemon reachable
[✓] Policy server reachable at http://localhost:8080/v1

# Configuration:
#   POLICY_URL       = http://localhost:8080/v1
#   POLICY_MODEL     = Qwen3-8B-4bit
#   MAX_CONCURRENT   = 4
#   N_SAMPLES        = 8
#   ROLLOUT_BATCH    = 4
#   MAX_TURNS        = 20
#   DATASET          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/data/sample_tasks.jsonl
#   LOG_DIR          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/logs
