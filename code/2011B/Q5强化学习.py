import gym
from gym import spaces
import numpy as np
import random
import networkx as nx
from collections import defaultdict

class PoliceThiefEnv(gym.Env):
    def __init__(self, city_graph, police_stations, thief_start):
        super(PoliceThiefEnv, self).__init__()
        self.city_graph = city_graph
        self.police_stations = police_stations
        self.thief_start = thief_start
        self.num_police = len(police_stations)

        self.action_space = spaces.MultiDiscrete([len(city_graph.nodes) for _ in range(self.num_police)])
        self.observation_space = spaces.Box(low=0, high=len(city_graph.nodes)-1, shape=(self.num_police + 1,), dtype=np.int32)

        self.reset()

    def reset(self):
        self.police_positions = list(self.police_stations)
        self.thief_position = self.thief_start
        return np.array(self.police_positions + [self.thief_position], dtype=np.int32)

    def step(self, actions):
        # Update police positions
        for i, action in enumerate(actions):
            if action in self.city_graph[self.police_positions[i]]:
                self.police_positions[i] = action

        # Update thief position (random move for simplicity)
        self.thief_position = random.choice(list(self.city_graph[self.thief_position]))

        done = any(police == self.thief_position for police in self.police_positions)
        reward = 1 if done else -0.1
        return np.array(self.police_positions + [self.thief_position], dtype=np.int32), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((len(self.city_graph.nodes), len(self.city_graph.nodes)))
        for pos in self.police_positions:
            grid[pos] = 1
        grid[self.thief_position] = 2
        print(grid)

# 创建城市的道路网络
city_graph = nx.Graph()

# 定义路口位置和道路长度
intersections = {
    0: (0, 0),
    1: (2, 4),
    2: (5, 5),
    3: (7, 8),
    4: (9, 2),
    5: (10, 4),
    6: (8, 6)
}

roads = [
    (0, 1, 5),
    (0, 2, 9),
    (1, 2, 2),
    (1, 3, 7),
    (2, 3, 3),
    (2, 4, 6),
    (3, 4, 4),
    (3, 5, 5),
    (4, 5, 2),
    (4, 6, 3),
    (5, 6, 1)
]

# 添加节点和边到图中
for node, position in intersections.items():
    city_graph.add_node(node, pos=position)

for road in roads:
    city_graph.add_edge(road[0], road[1], weight=road[2])

# 定义警察局和小偷的初始位置
police_stations = [0, 1, 2, 3]
thief_start = 6

# 创建环境
env = PoliceThiefEnv(city_graph, police_stations, thief_start)

# 测试环境
obs = env.reset()
print("Initial State:", obs)
env.render()

# 定义 Q-learning 智能体
class QLearningAgent:
    def __init__(self, n_actions, n_agents, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros((n_agents, n_actions)))

    def choose_actions(self, state):
        actions = []
        for agent_id in range(self.n_agents):
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(range(self.n_actions))
            else:
                action = np.argmax(self.q_table[state][agent_id])
            actions.append(action)
        return actions

    def update_q_table(self, state, actions, reward, next_state):
        for agent_id in range(self.n_agents):
            best_next_action = np.argmax(self.q_table[next_state][agent_id])
            td_target = reward + self.gamma * self.q_table[next_state][agent_id][best_next_action]
            td_error = td_target - self.q_table[state][agent_id][actions[agent_id]]
            self.q_table[state][agent_id][actions[agent_id]] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

# 创建 Q-learning 智能体
n_actions = env.action_space.nvec[0]
n_agents = len(police_stations)
agent = QLearningAgent(n_actions, n_agents)

# 训练智能体
n_episodes = 1000
for episode in range(n_episodes):
    state = tuple(env.reset())
    done = False
    while not done:
        actions = agent.choose_actions(state)
        next_state, reward, done, _ = env.step(actions)
        next_state = tuple(next_state)
        agent.update_q_table(state, actions, reward, next_state)
        state = next_state
        agent.decay_epsilon()

# 测试训练好的智能体
state = tuple(env.reset())
done = False
while not done:
    actions = agent.choose_actions(state)
    next_state, reward, done, _ = env.step(actions)
    env.render()
    state = tuple(next_state)
    if done:
        print("Caught the thief!")
