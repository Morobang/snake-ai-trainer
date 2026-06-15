"""
train.py — Training Loop with Live Pygame Visualizer
======================================================
Run this file to train the DQN agent and WATCH it learn in real time.

A pygame window opens showing:
  - The snake playing live
  - Stats panel: episode, score, best, deaths, epsilon
  - Score chart updating per run
  - Speed slider to control training pace

Controls:
  SPACE     — pause / resume
  UP/DOWN   — speed up / slow down
  R         — reset AI and start fresh
  Q / ESC   — quit and save

Usage:
  python3 model/train.py                  # train with visual (default)
  python3 model/train.py --no-visual      # headless, terminal only
  python3 model/train.py --episodes 2000  # more episodes
  python3 model/train.py --resume         # continue from saved weights
"""

import sys
import os
import argparse
import csv
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from game  import SnakeEnv, UP, DOWN, LEFT, RIGHT
from agent import DQNAgent


# ── CONSTANTS ────────────────────────────────────────────────────────────────

GRID      = 20
CELL      = 24          # px per cell
PANEL_W   = 320         # right stats panel width
WIN_W     = GRID * CELL + PANEL_W
WIN_H     = GRID * CELL

# colours
C_BG      = ( 10,  10,  15)
C_PANEL   = ( 17,  17,  24)
C_BORDER  = ( 30,  30,  46)
C_GREEN   = (  0, 255, 136)
C_GREEN2  = (  0, 180,  90)
C_RED     = (255,  68, 102)
C_YELLOW  = (255, 204,   0)
C_BLUE    = ( 68, 136, 255)
C_TEXT    = (200, 200, 216)
C_DIM     = (100, 100, 128)
C_FOOD    = (255, 204,   0)
C_HEAD    = (255, 255, 255)
C_GRID    = ( 15,  15,  26)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Snake DQN with live visualizer')
    p.add_argument('--episodes',   type=int,  default=1000)
    p.add_argument('--no-visual',  action='store_true', help='Headless training, no window')
    p.add_argument('--resume',     action='store_true', help='Resume from saved weights')
    p.add_argument('--speed',      type=int,  default=10,  help='Initial render speed (1-60 fps)')
    return p.parse_args()


# ── CSV LOGGING ──────────────────────────────────────────────────────────────

def init_csv(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'episode','score','steps','reward',
            'epsilon','loss','memory_size',
            'avg_score_10','avg_score_100','best_score','duration_s'
        ])

def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


# ── PYGAME RENDERER ──────────────────────────────────────────────────────────

class Visualizer:
    def __init__(self, episodes_total):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption('Snake AI Trainer — DQN')

        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont('Courier New', 22, bold=True)
        self.font_md = pygame.font.SysFont('Courier New', 14, bold=True)
        self.font_sm = pygame.font.SysFont('Courier New', 11)

        self.episodes_total = episodes_total
        self.score_history  = []
        self.paused         = False
        self.fps            = 10       # render speed
        self.running        = True

    def handle_events(self):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN:
                if event.key in (pg.K_q, pg.K_ESCAPE):
                    self.running = False
                if event.key == pg.K_SPACE:
                    self.paused = not self.paused
                if event.key == pg.K_UP:
                    self.fps = min(120, self.fps + 5)
                if event.key == pg.K_DOWN:
                    self.fps = max(1, self.fps - 5)

    def draw(self, env, episode, best_score, total_deaths, avg10):
        pg     = self.pygame
        screen = self.screen
        screen.fill(C_BG)

        self._draw_grid()
        self._draw_food(env.food)
        self._draw_snake(env.snake)
        self._draw_panel(episode, env.score, best_score,
                         total_deaths, avg10, env)
        self._draw_chart()
        self._draw_controls()

        pg.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self):
        pg = self.pygame
        for i in range(GRID + 1):
            pg.draw.line(self.screen, C_GRID,
                         (i * CELL, 0), (i * CELL, WIN_H))
            pg.draw.line(self.screen, C_GRID,
                         (0, i * CELL), (GRID * CELL, i * CELL))

    def _draw_food(self, food):
        pg  = self.pygame
        cx  = food.x * CELL + CELL // 2
        cy  = food.y * CELL + CELL // 2
        pg.draw.circle(self.screen, C_FOOD, (cx, cy), CELL // 3)
        # glow ring
        pg.draw.circle(self.screen, (*C_FOOD, 60), (cx, cy), CELL // 2, 2)

    def _draw_snake(self, snake):
        pg = self.pygame
        for i, p in enumerate(snake):
            rect = (p.x * CELL + 1, p.y * CELL + 1, CELL - 2, CELL - 2)
            if i == 0:
                color = C_HEAD
            elif i == 1:
                color = C_GREEN
            else:
                fade = max(40, 180 - i * 4)
                color = (0, fade, int(fade * 0.53))
            pg.draw.rect(self.screen, color, rect, border_radius=3)

    def _draw_panel(self, episode, score, best_score, deaths, avg10, env):
        pg     = self.pygame
        screen = self.screen
        px     = GRID * CELL   # panel x start

        # panel background
        pg.draw.rect(screen, C_PANEL, (px, 0, PANEL_W, WIN_H))
        pg.draw.line(screen, C_BORDER, (px, 0), (px, WIN_H), 1)

        x  = px + 16
        y  = 16

        def label(text, color=C_DIM, size='sm'):
            f = self.font_sm if size == 'sm' else self.font_md
            s = f.render(text, True, color)
            screen.blit(s, (x, y))

        def big(text, color=C_GREEN):
            s = self.font_lg.render(text, True, color)
            screen.blit(s, (x, y))

        def divider():
            nonlocal y
            pg.draw.line(screen, C_BORDER, (px + 8, y + 6), (px + PANEL_W - 8, y + 6), 1)
            y += 16

        # title
        label('SNAKE AI TRAINER', C_GREEN, 'md')
        y += 20
        label(f'DQN  |  {GRID}×{GRID} grid', C_DIM)
        y += 20
        divider()

        # episode
        label('EPISODE', C_DIM)
        y += 16
        big(f'{episode} / {self.episodes_total}', C_TEXT)
        y += 28

        # score row
        label('SCORE  THIS RUN', C_DIM)
        y += 16
        big(str(score), C_GREEN)
        y += 28

        label('BEST EVER', C_DIM)
        y += 16
        big(str(best_score), C_YELLOW)
        y += 28

        label('AVG LAST 10', C_DIM)
        y += 16
        big(f'{avg10:.1f}', C_BLUE)
        y += 28

        label('TOTAL DEATHS', C_DIM)
        y += 16
        big(str(deaths), C_RED)
        y += 28

        divider()

        # epsilon bar
        label('EXPLORATION  (epsilon)', C_DIM)
        y += 16
        eps     = env.__dict__.get('_epsilon_display', 1.0)
        bar_w   = PANEL_W - 32
        bar_h   = 8
        pg.draw.rect(screen, C_BORDER, (x, y, bar_w, bar_h), border_radius=4)
        fill_w  = int(bar_w * eps)
        if fill_w > 0:
            pg.draw.rect(screen, C_YELLOW, (x, y, fill_w, bar_h), border_radius=4)
        y += 20
        label(f'{eps*100:.1f}%  exploring  →  {(1-eps)*100:.1f}%  exploiting', C_DIM)
        y += 20

        divider()

        # status
        if self.paused:
            label('[ PAUSED — SPACE to resume ]', C_YELLOW, 'md')
        else:
            label(f'SPEED: {self.fps} fps  (↑↓ to adjust)', C_DIM)
        y += 18

    def _draw_chart(self):
        """Score-per-run bar chart in the bottom of the panel."""
        pg     = self.pygame
        screen = self.screen
        px     = GRID * CELL
        chart_y = WIN_H - 120
        chart_h = 100
        chart_w = PANEL_W - 32
        x       = px + 16

        pg.draw.line(screen, C_BORDER,
                     (px + 8, chart_y - 20), (px + PANEL_W - 8, chart_y - 20), 1)
        label_s = self.font_sm.render('SCORE PER RUN  (last 60)', True, C_DIM)
        screen.blit(label_s, (x, chart_y - 18))

        data = self.score_history[-60:]
        if len(data) < 2:
            return

        max_s = max(data) if max(data) > 0 else 1
        bar_w = max(1, chart_w // len(data))

        for i, s in enumerate(data):
            bh    = int((s / max_s) * (chart_h - 10))
            bx    = x + i * bar_w
            by    = chart_y + chart_h - bh
            alpha = 100 + int(155 * (i / len(data)))
            color = (0, alpha, int(alpha * 0.53))
            if i == len(data) - 1:
                color = C_GREEN
            pg.draw.rect(screen, color, (bx, by, max(1, bar_w - 1), bh))

    def _draw_controls(self):
        pg     = self.pygame
        screen = self.screen
        hints  = [
            'SPACE  pause/resume',
            '↑ ↓    speed',
            'Q/ESC  quit & save',
        ]
        y = WIN_H - 46
        for h in hints:
            s = self.font_sm.render(h, True, C_DIM)
            screen.blit(s, (GRID * CELL + 16, y))
            y += 14

    def tick_paused(self):
        """Keep window alive and handle events while paused."""
        self.handle_events()
        self.clock.tick(30)

    def close(self):
        self.pygame.quit()


# ── HEADLESS LOGGING (no visual) ─────────────────────────────────────────────

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
    print(f"   Grid     : {GRID}×{GRID}")
    print(f"   Visual   : {'No' if args.no_visual else 'Yes — window opening...'}")

    weights_path = 'model/weights.pth'
    json_path    = 'model/weights.json'
    log_path     = 'results/training_log.csv'

    env   = SnakeEnv(grid_size=GRID)
    agent = DQNAgent()

    if args.resume and Path(weights_path).exists():
        agent.load(weights_path)
        agent.epsilon = 0.1
        print(f"   Resumed from {weights_path}\n")

    # attach epsilon to env so visualizer can read it
    env._epsilon_display = agent.epsilon

    vis = None
    if not args.no_visual:
        vis = Visualizer(args.episodes)

    init_csv(log_path)

    score_history = []
    best_score    = 0
    total_deaths  = 0
    start_time    = time.time()

    if args.no_visual:
        print_header()

    # ── EPISODE LOOP ──────────────────────────────────────────────────────────
    for episode in range(1, args.episodes + 1):

        # check if window was closed
        if vis and not vis.running:
            print("\nWindow closed — saving and exiting.")
            break

        state        = env.reset()
        done         = False
        total_reward = 0.0
        ep_losses    = []
        ep_start     = time.time()

        # ── STEP LOOP ─────────────────────────────────────────────────────────
        while not done:

            # handle pause
            if vis:
                while vis.paused and vis.running:
                    vis.tick_paused()
                if not vis.running:
                    break

                vis.handle_events()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)

            state         = next_state
            total_reward += reward

            # render every step
            if vis:
                avg10 = float(np.mean(score_history[-10:])) if score_history else 0
                env._epsilon_display = agent.epsilon
                vis.draw(env, episode, best_score, total_deaths, avg10)

        # ── END OF EPISODE ────────────────────────────────────────────────────
        score = info['score']
        steps = info['steps']
        score_history.append(score)
        if vis:
            vis.score_history = score_history

        if info['reason'] in ('collision', 'loop'):
            total_deaths += 1

        agent.decay_epsilon()
        if episode % agent.target_update == 0:
            agent.sync_target()

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        avg10    = float(np.mean(score_history[-10:]))
        avg100   = float(np.mean(score_history[-100:]))
        new_best = score > best_score

        if new_best:
            best_score = score
            agent.save(weights_path)

        append_csv(log_path, [
            episode, score, steps, round(total_reward, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            agent.memory_size,
            round(avg10, 2), round(avg100, 2),
            best_score, round(time.time() - ep_start, 3)
        ])

        if args.no_visual and (episode % 10 == 0 or new_best):
            print_row(episode, score, steps, total_reward,
                      agent.epsilon, avg_loss, avg10, best_score, new_best)

    # ── DONE ──────────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n✓ Training complete in {total_time:.1f}s")
    print(f"  Best score    : {best_score}")
    print(f"  Total episodes: {len(score_history)}")

    agent.save(weights_path)
    agent.export_json(json_path)
    print(f"  Weights saved → {weights_path}")
    print(f"  Browser JSON  → {json_path}")
    print(f"  Log saved     → {log_path}")

    if vis:
        vis.close()

    return score_history, best_score


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Weights saved.")
