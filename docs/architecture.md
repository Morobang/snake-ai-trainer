# Architecture: Snake AI Trainer

## Overview

Single-file browser app (`index.html`) — no build step, no backend, no dependencies.

---

## Game Engine

- **Grid:** 20x20 cells rendered on an HTML5 Canvas
- **Tick rate:** Configurable (default 150ms, slider goes from 10ms to 500ms)
- **State:** Snake body array (head at index 0), food position, direction, score, move count, time elapsed
- **Collision:** Wall and self-collision end the game

---

## Q-Learning Agent

### State Representation (11 binary inputs)

| Signal | Description |
|--------|-------------|
| danger_straight | Wall or body directly ahead |
| danger_left | Wall or body to the left |
| danger_right | Wall or body to the right |
| dir_up | Currently moving up |
| dir_down | Currently moving down |
| dir_left | Currently moving left |
| dir_right | Currently moving right |
| food_up | Food is above head |
| food_down | Food is below head |
| food_left | Food is to the left |
| food_right | Food is to the right |

Total possible states: 2^11 = **2048**

### Q-Table

- Shape: `2048 x 4` (states x actions)
- Actions: UP, DOWN, LEFT, RIGHT
- Initialised to zero
- Persisted to `localStorage` so training survives page refresh

### Update Rule (Bellman Equation)

```
Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

- **α (learning rate):** 0.1
- **γ (discount factor):** 0.9
- **ε (exploration rate):** Starts at 1.0, decays to 0.01 over training

### Reward Structure

| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Die | -10 |
| Move toward food | +1 |
| Move away from food | -1 |

---

## Visualization Layer

- **Live canvas render** of AI gameplay during training
- **Score-per-run chart** using Canvas 2D API (no chart library)
- **Death heatmap** — 20x20 grid tracking death frequency per cell
- **Stats panel** — current run score, best score, total deaths, moves this run
- **Speed slider** — controls game tick interval

---

## Modes

| Mode | Description |
|------|-------------|
| Player | Human plays with arrow keys |
| Train | AI trains continuously, visualized live |
| Race | Human and AI play simultaneously side by side |

---

## Persistence

- Q-table saved to `localStorage` on each episode end
- Player personal best saved to `localStorage`
- Reset button clears both

