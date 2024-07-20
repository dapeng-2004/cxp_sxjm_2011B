import networkx as nx

# 创建图
G = nx.Graph()
# 添加节点和边
G.add_edges_from([
    (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)
])

# 初始化小偷位置和警察位置
thief_position = 1
police_positions = [2, 3]

# 模拟小偷逃跑三分钟
thief_possible_positions = {thief_position}
for _ in range(3):
    new_positions = set()
    for pos in thief_possible_positions:
        new_positions.update(G.neighbors(pos))
    thief_possible_positions.update(new_positions)

# 封锁策略
def block_path(G, path):
    G.remove_edges_from(path)

# 示例封锁路径
block_path(G, [(1, 2)])

# 更新小偷可能位置
new_positions = set()
for pos in thief_possible_positions:
    new_positions.update(G.neighbors(pos))
thief_possible_positions.update(new_positions)

print("小偷可能的位置:", thief_possible_positions)
