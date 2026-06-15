# 🐍 Snake AI Trainer

**[▶ Play it live →](https://morobang.github.io/snake-ai-trainer/)**

A browser-based Snake game with a **Q-learning AI agent** that trains in real time. Watch it fail hundreds of times, then slowly figure out how to survive — no backend, no install, just open the link.

---

## What This Is

Two things in one:

1. **A playable Snake game** — keyboard controls, move counter, timer, high score tracking
2. **A live AI training visualizer** — watch the Q-learning agent play, die, and improve over thousands of runs

The AI uses a **Q-table** (not a neural net) so training is fast enough to watch in real time in the browser. You can control the speed, see the score chart update per run, and compare your personal best against the trained agent.

---

## Features

- 🎮 Play Snake yourself with full stats (moves, time, score)
- 🤖 Watch the AI train live — see it fail and improve in real time
- 📈 Score-per-run chart so improvement is visible
- 🗺️ Death heatmap — see where on the grid the AI struggles most
- 👻 Ghost replay of the AI's best run ever
- ⚡ Speed slider — slow-mo to watch decisions, 100x to train fast
- 🏆 You vs AI mode — race the trained agent after training
- 💾 Q-table persists in localStorage — training survives page refresh

---

## How The AI Works (Plain English)

The agent sees 11 signals about its current situation:
- Is there danger directly ahead / left / right?
- Which direction is it moving?
- Which direction is the food?

It keeps a table of "how good is each action in each situation?" — starting at zero (knows nothing). Every time it eats food it gets rewarded. Every time it dies it gets penalised. Over thousands of games the table fills up and the snake gets smarter.

This is called **Q-learning** — a classic reinforcement learning algorithm.

---

## Project Structure

```
snake-ai-trainer/
├── index.html          ← entire game + AI, single file
├── docs/
│   └── architecture.md ← technical deep-dive on the Q-learning setup
└── README.md
```

---

## Run It Locally

Just clone and open `index.html` in any modern browser. No server needed.

```bash
git clone https://github.com/Morobang/snake-ai-trainer.git
cd snake-ai-trainer
open index.html
```

---

## Roadmap

- [x] Repo setup
- [x] Phase 1 — Playable Snake game with stats
- [x] Phase 2 — Q-learning agent + training loop
- [x] Phase 3 — Live visualizer (stats panel, chart, heatmap)
- [x] Phase 4 — You vs AI race mode

---

## Built By

**Morobang Tshigidimisa** (Rocket) — Data Engineer & ML Consultant, Pretoria 🇿🇦

> Portfolio project demonstrating reinforcement learning concepts through interactive browser-based visualization.

