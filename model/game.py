"""
game.py — Snake Environment
============================
This is the Snake game as a proper RL environment.
Pattern: reset() to start, step(action) to move, done when dead.

Same structure as OpenAI Gym — if you've ever seen:
    obs = env.reset()
    obs, reward, done, info = env.step(action)
...this is exactly that, hand-rolled for Snake.

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
"""

import random
import numpy as np
from collections import namedtuple

# A single point on the grid
Point = namedtuple('Point', ['x', 'y'])

# Action constants — use these everywhere, never raw ints
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

# Direction vectors: action → (dx, dy)
DIRECTION = {
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
}

# Rewards
REWARD_EAT   =  10
REWARD_DIE   = -10
REWARD_CLOSER =  1
REWARD_FARTHER = -1
REWARD_LOOP  =  -5   # penalise going too long without food


class SnakeEnv:
    """
    Snake game environment.

    Usage:
        env = SnakeEnv(grid_size=20)
        state = env.reset()
        state, reward, done, info = env.step(action)
    """

    def __init__(self, grid_size: int = 20, max_steps_without_food: int = None):
        self.grid_size = grid_size
        # if the snake goes this many steps without eating, episode ends
        self.max_steps_without_food = max_steps_without_food or grid_size * grid_size * 2
        self.reset()

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Start a new episode. Returns the initial state as a numpy array.
        Call this at the beginning of every episode.
        """
        cx = self.grid_size // 2
        cy = self.grid_size // 2

        # Snake starts as 3 cells, head at center, body going down
        self.snake = [
            Point(cx,   cy),
            Point(cx,   cy + 1),
            Point(cx,   cy + 2),
        ]

        self.direction  = UP
        self.score      = 0
        self.steps      = 0
        self.steps_since_food = 0
        self.food       = self._spawn_food()

        return self._get_state()

    def step(self, action: int):
        """
        Take one action. Returns (state, reward, done, info).

        Args:
            action: int — one of UP(0), DOWN(1), LEFT(2), RIGHT(3)

        Returns:
            state  : np.ndarray — 11 binary values describing the new situation
            reward : float      — what happened as a result of this action
            done   : bool       — True if the game is over
            info   : dict       — extra info (score, steps) for logging
        """
        self.steps += 1
        self.steps_since_food += 1

        # prevent 180-degree turns — ignore if trying to reverse
        action = self._filter_reverse(action)
        self.direction = action

        # move
        dx, dy = DIRECTION[action]
        new_head = Point(self.snake[0].x + dx, self.snake[0].y + dy)

        # check death
        dead = self._is_collision(new_head)

        if dead:
            reward = REWARD_DIE
            done   = True
            info   = {'score': self.score, 'steps': self.steps, 'reason': 'collision'}
            return self._get_state(), reward, done, info

        # check loop (too long without food)
        if self.steps_since_food >= self.max_steps_without_food:
            reward = REWARD_LOOP
            done   = True
            info   = {'score': self.score, 'steps': self.steps, 'reason': 'loop'}
            return self._get_state(), reward, done, info

        # distance to food before move (for shaping reward)
        dist_before = self._manhattan(self.snake[0], self.food)

        # move snake
        self.snake.insert(0, new_head)
        ate = (new_head == self.food)

        if ate:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            reward = REWARD_EAT
        else:
            self.snake.pop()  # remove tail if no food eaten
            dist_after = self._manhattan(new_head, self.food)
            reward = REWARD_CLOSER if dist_after < dist_before else REWARD_FARTHER

        done = False
        info = {'score': self.score, 'steps': self.steps, 'reason': None}
        return self._get_state(), reward, done, info

    @property
    def head(self) -> Point:
        return self.snake[0]

    # ── STATE REPRESENTATION ────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Encode the current game situation as 11 binary values.
        This is what the neural network sees as input.

        [0]  danger straight ahead
        [1]  danger to the left
        [2]  danger to the right
        [3]  currently moving up
        [4]  currently moving down
        [5]  currently moving left
        [6]  currently moving right
        [7]  food is above
        [8]  food is below
        [9]  food is to the left
        [10] food is to the right
        """
        head = self.head

        # what counts as "left" and "right" depends on current direction
        straight, left, right = self._relative_directions()

        state = [
            # danger signals
            int(self._is_collision(Point(head.x + straight[0], head.y + straight[1]))),
            int(self._is_collision(Point(head.x + left[0],     head.y + left[1]))),
            int(self._is_collision(Point(head.x + right[0],    head.y + right[1]))),

            # current direction (one-hot)
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),

            # food location relative to head
            int(self.food.y < head.y),   # food up
            int(self.food.y > head.y),   # food down
            int(self.food.x < head.x),   # food left
            int(self.food.x > head.x),   # food right
        ]

        return np.array(state, dtype=np.float32)

    # ── HELPERS ─────────────────────────────────────────────────────────────

    def _spawn_food(self) -> Point:
        """Place food somewhere that isn't occupied by the snake."""
        while True:
            f = Point(
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if f not in self.snake:
                return f

    def _is_collision(self, point: Point) -> bool:
        """True if this point is a wall or the snake's own body."""
        # wall
        if point.x < 0 or point.x >= self.grid_size:
            return True
        if point.y < 0 or point.y >= self.grid_size:
            return True
        # self
        if point in self.snake[1:]:
            return True
        return False

    def _filter_reverse(self, action: int) -> int:
        """Block 180-degree turns — return current direction instead."""
        opposites = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action == opposites.get(self.direction):
            return self.direction
        return action

    def _relative_directions(self):
        """
        Return (straight, left, right) as (dx,dy) tuples
        relative to the snake's current movement direction.
        """
        d = self.direction
        if d == UP:
            return (0,-1), (-1,0), (1,0)
        if d == DOWN:
            return (0,1),  (1,0),  (-1,0)
        if d == LEFT:
            return (-1,0), (0,1),  (0,-1)
        if d == RIGHT:
            return (1,0),  (0,-1), (0,1)

    def _manhattan(self, a: Point, b: Point) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    # ── DEBUG ────────────────────────────────────────────────────────────────

    def render_ascii(self):
        """Print the current board to terminal. Useful for debugging."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i, p in enumerate(self.snake):
            if 0 <= p.x < self.grid_size and 0 <= p.y < self.grid_size:
                grid[p.y][p.x] = 'O' if i == 0 else '#'

        if 0 <= self.food.x < self.grid_size and 0 <= self.food.y < self.grid_size:
            grid[self.food.y][self.food.x] = '*'

        border = '+' + '-' * self.grid_size + '+'
        print(border)
        for row in grid:
            print('|' + ''.join(row) + '|')
        print(border)
        print(f'Score: {self.score}  Steps: {self.steps}  Direction: {["UP","DOWN","LEFT","RIGHT"][self.direction]}')
        print()


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Run this file directly to verify the environment works.
    It plays 3 random episodes and prints what happens.
    """
    print("Testing SnakeEnv...\n")

    env = SnakeEnv(grid_size=10)

    for episode in range(3):
        state = env.reset()
        done  = False
        total_reward = 0
        steps = 0

        print(f"── Episode {episode + 1} ──")
        env.render_ascii()

        while not done:
            action = random.randint(0, 3)  # random agent
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Done! Score={info['score']}  Steps={steps}  "
              f"Total reward={total_reward:.1f}  Reason={info['reason']}\n")

    print("✓ Environment works correctly.")
