"""
agent.py — The DQN Agent
==========================
This is where the actual learning happens.

The agent sits between the environment and the network.
It decides what action to take, stores memories, and trains the network.

Two key ideas that make DQN work (and why they matter):

1. EXPERIENCE REPLAY
   Problem: if you train the network on every single step in order,
   each update is highly correlated with the last one — the network
   just chases its tail and never stabilises.
   Fix: store every (state, action, reward, next_state, done) in a
   memory buffer. Sample RANDOM batches from it to train on.
   Now each training step sees a mix of old and new experiences.

2. TARGET NETWORK
   Problem: the Q-value you're predicting AND the target you're
   training toward are both produced by the same network. Every time
   the network updates, the target shifts — like trying to hit a
   moving bullseye.
   Fix: keep TWO identical networks. The online network trains every
   step. The target network is a frozen copy that only updates every
   N steps. Stable target = stable training.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN
from game import UP, DOWN, LEFT, RIGHT


# ── REPLAY BUFFER ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    The agent's memory.

    Stores (state, action, reward, next_state, done) tuples — called
    "experiences" or "transitions". When full, old memories are dropped.

    Why deque? It's a double-ended queue. When maxlen is reached,
    appending a new item automatically drops the oldest one.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Pull a random batch of experiences.
        Returns everything as numpy arrays, ready to convert to tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN AGENT ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    The agent that learns to play Snake using Deep Q-Learning.

    Key components:
      - online_net   : the network being trained every step
      - target_net   : frozen copy, updated every TARGET_UPDATE steps
      - memory       : replay buffer
      - epsilon      : exploration rate (starts high, decays over time)
    """

    def __init__(
        self,
        state_size:    int   = 11,
        action_size:   int   = 4,
        hidden_size:   int   = 256,
        lr:            float = 0.001,      # learning rate
        gamma:         float = 0.9,        # discount factor
        epsilon:       float = 1.0,        # starting exploration rate
        epsilon_min:   float = 0.01,       # floor — always explore a little
        epsilon_decay: float = 0.995,      # how fast epsilon falls
        batch_size:    int   = 64,         # experiences per training step
        memory_size:   int   = 100_000,    # max memories stored
        target_update: int   = 100,        # sync target net every N episodes
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        # device — use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # two networks — same architecture, different roles
        self.online_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net never trains directly

        # Adam optimizer — standard choice for DQN
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Huber loss — less sensitive to outliers than MSE
        # when a reward is very large/small, MSE squares the error and
        # can cause huge destabilising gradient updates. Huber clips it.
        self.loss_fn = nn.SmoothL1Loss()

        # memory
        self.memory = ReplayBuffer(memory_size)

        # counters
        self.steps_done  = 0   # total steps across all episodes
        self.episode     = 0   # total episodes trained

    # ── ACTION SELECTION ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon  → explore: pick a random action
        With probability 1-epsilon → exploit: pick the network's best action

        Early in training epsilon is high (lots of exploration).
        Over time it decays so the agent trusts its learned knowledge more.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ── MEMORY ───────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        """Store one experience in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ── TRAINING STEP ────────────────────────────────────────────────────────

    def train_step(self) -> float:
        """
        One training step — sample a batch and update the online network.

        Returns the loss value for logging. Returns 0 if not enough
        memories yet to fill a batch.

        The Bellman equation update:
            Q(s, a) = r + gamma * max(Q_target(s', a'))

        In plain English:
            "The value of being in state s and taking action a
             equals the reward I got, plus the discounted value
             of the best action I can take from the next state."
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # sample random batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # convert to tensors
        states_t      = torch.tensor(states,      dtype=torch.float32).to(self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.int64).to(self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        # ── current Q-values ──
        # online_net predicts Q for all 4 actions
        # we gather only the Q-value for the action that was actually taken
        self.online_net.train()
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── target Q-values ──
        # target_net predicts Q for the next state
        # if episode ended (done=1), there is no next state — target is just r
        with torch.no_grad():
            next_q    = self.target_net(next_states_t).max(1)[0]
            target_q  = rewards_t + self.gamma * next_q * (1 - dones_t)

        # ── compute loss and backpropagate ──
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.steps_done += 1

        return float(loss.item())

    # ── EPSILON DECAY ────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Call at the end of each episode to reduce exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    # ── TARGET NETWORK SYNC ──────────────────────────────────────────────────

    def sync_target(self):
        """
        Copy online network weights into target network.
        Called every TARGET_UPDATE episodes.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── SAVE / LOAD ──────────────────────────────────────────────────────────

    def save(self, path: str = 'model/weights.pth'):
        """Save the online network weights."""
        self.online_net.save(path)

    def load(self, path: str = 'model/weights.pth'):
        """Load weights into both networks."""
        self.online_net.load(path)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def export_json(self, path: str = 'model/weights.json'):
        """Export weights as JSON for the browser."""
        self.online_net.export_json(path)

    @property
    def memory_size(self):
        return len(self.memory)


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from game import SnakeEnv

    print("Testing DQNAgent...\n")

    env   = SnakeEnv(grid_size=10)
    agent = DQNAgent()

    print(f"Running 5 episodes with random + learned actions...\n")

    for ep in range(5):
        state = env.reset()
        done  = False
        total_reward = 0
        losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        if (ep + 1) % agent.target_update == 0:
            agent.sync_target()

        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode {ep+1:3d} | Score: {info['score']:3d} | "
              f"Reward: {total_reward:7.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Memory: {agent.memory_size:5d} | "
              f"Loss: {avg_loss:.4f}")

    print(f"\n✓ DQNAgent works correctly.")
