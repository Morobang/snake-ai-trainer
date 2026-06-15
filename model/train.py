"""
train.py — Training Loop
==========================
This is the script you actually run to train the agent.

It ties everything together:
  game.py   → the environment
  model.py  → the neural network
  agent.py  → the learning logic

What happens here:
  1. Run episode after episode of Snake
  2. After each episode, train the network on a batch from memory
  3. Every 100 episodes, sync the target network
  4. Log everything to results/training_log.csv
  5. Save the best model whenever a new high score is hit
  6. Export weights.json at the end for the browser

Usage:
  python3 model/train.py                        # train from scratch
  python3 model/train.py --episodes 2000        # custom episode count
  python3 model/train.py --resume               # continue from saved weights
  python3 model/train.py --episodes 500 --grid 10  # smaller grid, faster training
"""

import sys
import os
import argparse
import csv
import time
import numpy as np
from pathlib import Path

# make sure imports work when run from repo root or model/ folder
sys.path.insert(0, str(Path(__file__).parent))

from game  import SnakeEnv
from agent import DQNAgent


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train the Snake DQN agent')
    p.add_argument('--episodes',  type=int,   default=1000,  help='Number of training episodes')
    p.add_argument('--grid',      type=int,   default=20,    help='Grid size (default 20)')
    p.add_argument('--resume',    action='store_true',       help='Resume from saved weights')
    p.add_argument('--no-save',   action='store_true',       help='Skip saving weights')
    p.add_argument('--log-every', type=int,   default=10,    help='Print progress every N episodes')
    return p.parse_args()


# ── LOGGING ──────────────────────────────────────────────────────────────────

def init_csv(path: str):
    """Create the CSV log file with headers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'score', 'steps', 'reward',
            'epsilon', 'loss', 'memory_size',
            'avg_score_10', 'avg_score_100', 'best_score', 'duration_s'
        ])

def append_csv(path: str, row: list):
    """Append one row to the CSV log."""
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ── PROGRESS PRINTING ────────────────────────────────────────────────────────

def print_header():
    print("\n" + "═" * 75)
    print(f"  {'EP':>6}  {'SCORE':>6}  {'STEPS':>6}  {'REWARD':>8}  "
          f"{'EPS':>6}  {'LOSS':>8}  {'AVG10':>6}  {'BEST':>6}")
    print("═" * 75)

def print_row(ep, score, steps, reward, eps, loss, avg10, best, new_best=False):
    marker = " ★" if new_best else ""
    print(f"  {ep:>6}  {score:>6}  {steps:>6}  {reward:>8.1f}  "
          f"{eps:>6.3f}  {loss:>8.4f}  {avg10:>6.1f}  {best:>6}{marker}")


# ── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def train(args):
    print(f"\n🐍 Snake DQN Training")
    print(f"   Episodes : {args.episodes}")
    print(f"   Grid     : {args.grid}×{args.grid}")
    print(f"   Resume   : {args.resume}")

    # paths
    weights_path = 'model/weights.pth'
    json_path    = 'model/weights.json'
    log_path     = 'results/training_log.csv'

    # init environment and agent
    env   = SnakeEnv(grid_size=args.grid)
    agent = DQNAgent(
        state_size    = 11,
        action_size   = 4,
        hidden_size   = 256,
        lr            = 0.001,
        gamma         = 0.9,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.995,
        batch_size    = 64,
        memory_size   = 100_000,
        target_update = 100,
    )

    # resume from checkpoint if requested
    if args.resume and Path(weights_path).exists():
        agent.load(weights_path)
        agent.epsilon = 0.1  # mostly exploit when resuming
        print(f"   Resumed from {weights_path} (epsilon reset to 0.1)\n")

    # init CSV log
    init_csv(log_path)

    # tracking
    score_history  = []
    best_score     = 0
    start_time     = time.time()
    total_loss     = 0.0
    loss_count     = 0

    print_header()

    # ── EPISODE LOOP ──────────────────────────────────────────────────────────
    for episode in range(1, args.episodes + 1):
        state        = env.reset()
        done         = False
        total_reward = 0.0
        ep_losses    = []
        ep_start     = time.time()

        # ── STEP LOOP ─────────────────────────────────────────────────────────
        while not done:
            # agent picks action
            action = agent.select_action(state)

            # environment responds
            next_state, reward, done, info = env.step(action)

            # store experience
            agent.remember(state, action, reward, next_state, done)

            # train on a random batch from memory
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)

            state         = next_state
            total_reward += reward

        # ── END OF EPISODE ────────────────────────────────────────────────────
        score = info['score']
        steps = info['steps']
        score_history.append(score)

        # decay exploration
        agent.decay_epsilon()

        # sync target network every 100 episodes
        if episode % agent.target_update == 0:
            agent.sync_target()

        # metrics
        avg_loss    = float(np.mean(ep_losses)) if ep_losses else 0.0
        avg10       = float(np.mean(score_history[-10:]))
        avg100      = float(np.mean(score_history[-100:]))
        ep_duration = time.time() - ep_start
        new_best    = score > best_score

        if new_best:
            best_score = score
            if not args.no_save:
                agent.save(weights_path)

        # log to CSV
        append_csv(log_path, [
            episode, score, steps, round(total_reward, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            agent.memory_size,
            round(avg10, 2), round(avg100, 2),
            best_score, round(ep_duration, 3)
        ])

        # print progress
        if episode % args.log_every == 0 or new_best:
            print_row(episode, score, steps, total_reward,
                      agent.epsilon, avg_loss, avg10, best_score, new_best)

    # ── TRAINING COMPLETE ─────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("═" * 75)
    print(f"\n✓ Training complete in {total_time:.1f}s")
    print(f"  Best score   : {best_score}")
    print(f"  Avg (last 100): {np.mean(score_history[-100:]):.2f}")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Log saved    → {log_path}")

    # export JSON for browser
    if not args.no_save:
        agent.export_json(json_path)
        print(f"  Weights saved → {weights_path}")
        print(f"  Browser JSON  → {json_path}")

    return score_history, best_score


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    try:
        score_history, best_score = train(args)

        # quick summary chart in terminal (ASCII)
        print("\n── Score over time (last 50 episodes) ──")
        last50 = score_history[-50:]
        max_s  = max(last50) if last50 else 1
        for i, s in enumerate(last50):
            bar = '█' * int((s / max_s) * 30) if max_s > 0 else ''
            print(f"  {i+len(score_history)-50+1:>4}  {bar:<30}  {s}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Weights saved to model/weights.pth")
