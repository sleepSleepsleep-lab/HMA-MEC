"""Train a new distilled policy using only K=8,M=4 samples from debate_dataset.jsonl."""
import os, sys, json, numpy as np
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from config import RESULTS_DIR, CHECKPOINT_DIR, NUM_USERS, NUM_EDGE_SERVERS
from distill_agent import DistillAgentTrainer, load_debate_dataset

DATA_PATH = os.path.join(RESULTS_DIR, "debate_dataset.jsonl")
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "distilled_policy.pth")

# Load only K=8 samples
states, alphas, servers, confs = [], [], [], []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line.strip())
        K = len(r['alpha'])
        if K != NUM_USERS: continue
        states.append(r['state']); alphas.append(r['alpha'])
        servers.append(r['server'])
        if r.get('confidence'): confs.append(r['confidence'])

states = np.array(states, dtype=np.float32)
alphas = np.array(alphas, dtype=np.float32)
servers = np.array(servers, dtype=int)
confs = np.array(confs, dtype=np.float32) if len(confs) == len(states) else None

print(f"Training on {len(states)} K=8 samples, state_dim={states.shape[1]}")
trainer = DistillAgentTrainer(
    K=NUM_USERS, M=NUM_EDGE_SERVERS,
    state_dim=states.shape[1],
    epochs=60, batch=128,
    save_path=SAVE_PATH)
trainer.train(states, alphas, servers, confidences=confs, val_ratio=0.1)
print(f"Model saved to {SAVE_PATH}")
