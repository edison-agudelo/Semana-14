import numpy as np
import pickle
import os

# =============================
# ENTORNO GRIDWORLD 5x5
# =============================

class GridWorldEnv:
    def __init__(self, size=5, obstacles=None, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []
        self.n_states = size * size
        self.n_actions = 4  # 0=arriba, 1=abajo, 2=izquierda, 3=derecha
        self.reset()

    def state_to_index(self, pos):
        r, c = pos
        return r * self.size + c

    def index_to_state(self, index):
        r = index // self.size
        c = index % self.size
        return (r, c)

    def reset(self):
        self.agent_pos = self.start
        return self.state_to_index(self.agent_pos)

    def step(self, action):
        r, c = self.agent_pos

        if action == 0:     # arriba
            new_pos = (r - 1, c)
        elif action == 1:   # abajo
            new_pos = (r + 1, c)
        elif action == 2:   # izquierda
            new_pos = (r, c - 1)
        else:               # derecha
            new_pos = (r, c + 1)

        if (
            new_pos[0] < 0 or new_pos[0] >= self.size or
            new_pos[1] < 0 or new_pos[1] >= self.size or
            new_pos in self.obstacles
        ):
            reward = -10
            new_pos = self.agent_pos
        else:
            reward = -1

        done = False

        if new_pos == self.goal:
            reward = 10
            done = True

        self.agent_pos = new_pos
        next_state = self.state_to_index(self.agent_pos)
        return next_state, reward, done


# =============================
# AGENTE Q-LEARNING
# =============================

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):

        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        q_current = self.Q[state, action]
        q_next = np.max(self.Q[next_state]) if not done else 0
        target = reward + self.gamma * q_next

        self.Q[state, action] += self.alpha * (target - q_current)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =============================
# ENTRENAMIENTO
# =============================

def train_agent(
    episodes=500,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    save_path="q_table.pkl"
):
    env = GridWorldEnv(
        size=5,
        obstacles=[(1,1), (2,3)],
        start=(0,0),
        goal=(4,4)
    )

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    rewards_per_episode = []
    epsilons = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        epsilons.append(agent.epsilon)

    with open(save_path, "wb") as f:
        pickle.dump(agent.Q, f)

    return rewards_per_episode, epsilons, agent


# =============================
# EJECUTAR EPISODIO GREEDY
# =============================

def load_q_table(path="q_table.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_greedy_episode(max_steps=50, q_path="q_table.pkl"):
    Q = load_q_table(q_path)
    if Q is None:
        return None

    env = GridWorldEnv(
        size=5,
        obstacles=[(1,1), (2,3)],
        start=(0,0),
        goal=(4,4)
    )

    state = env.reset()
    trajectory = []
    total_reward = 0

    for step in range(max_steps):
        action = int(np.argmax(Q[state]))
        row, col = env.index_to_state(state)

        trajectory.append({
            "step": step,
            "row": row,
            "col": col,
            "action": action
        })

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            row, col = env.index_to_state(state)
            trajectory.append({
                "step": step + 1,
                "row": row,
                "col": col,
                "action": None
            })
            break

    return {
        "trajectory": trajectory,
        "total_reward": total_reward
    }
