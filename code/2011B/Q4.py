import numpy as np
import Q5_Police_Assignment

# 假设数据已经被读取到以下变量中
n = 3  # 3 个人
m = 4  # 4 堆苹果
D = [
    [1, 2, 3, 4],
    [2, 4, 6, 8],
    [3, 6, 9, 12]
]
A = [10, 20, 30, 40]
alpha = 0.1  # 权重参数

# 创建线性规划问题
prob = pulp.LpProblem("AppleDistribution", pulp.LpMinimize)

# 创建决策变量
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(m)), lowBound=0, cat='Integer')

# 辅助变量，用于表示每个人的总代价与平均总代价的差异
deviation = pulp.LpVariable.dicts("deviation", (i for i in range(n)), lowBound=0)

# 计算每个人的总代价
total_cost = [pulp.lpSum(x[i, j] * D[i][j] for j in range(m)) + alpha * pulp.lpSum(x[i, j] for j in range(m)) for i in range(n)]

# 平均总代价
average_cost = pulp.lpSum(total_cost) / n

# 定义目标函数：最小化每个人总代价与平均总代价的差异的和
prob += pulp.lpSum(deviation[i] for i in range(n))

# 约束条件：每堆苹果都被分完
for j in range(m):
    prob += pulp.lpSum(x[i, j] for i in range(n)) == A[j]

# 约束条件：计算每个人的总代价与平均总代价的差异
for i in range(n):
    prob += deviation[i] >= total_cost[i] - average_cost
    prob += deviation[i] >= average_cost - total_cost[i]

# 解决问题
prob.solve()

# 输出结果
print("Status:", pulp.LpStatus[prob.status])
for i in range(n):
    for j in range(m):
        print(f"Person {i} takes {pulp.value(x[i, j])} apples from heap {j}")

# 计算每一个匹配对的时间和总成本
matching_pairs_with_time = []
total_durations = []
for i in range(n):
    total_cost_i = sum(pulp.value(x[i, j]) * D[i][j] for j in range(m)) + alpha * sum(pulp.value(x[i, j]) for j in range(m))
    total_durations.append(total_cost_i)
    for j in range(m):
        time = pulp.value(x[i, j]) * D[i][j]
        matching_pairs_with_time.append((i, j, pulp.value(x[i, j]), time))

print("匹配对及时间:")
for match in matching_pairs_with_time:
    if match[2] > 0:
        print(f"Person {match[0]} takes {match[2]} apples from heap {match[1]} with cost {match[3]}")

print("每个人的总成本:")
for i in range(n):
    print(f"Total cost for person {i}: {total_durations[i]}")
