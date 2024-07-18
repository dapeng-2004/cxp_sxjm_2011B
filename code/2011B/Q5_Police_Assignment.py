import pandas as pd
import numpy as np

# 手动创建地理位置数据框
data = {
    '全市路口节点标号': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92],
    '路口的横坐标X': [413, 403, 383.5, 381, 339, 335, 317, 334.5, 333, 282, 247, 219, 225, 280,
                 290, 337, 415, 432, 418, 444, 251, 234, 225, 212, 227, 256, 250.5, 243, 246,
                 314, 315, 326, 327, 328, 336, 336, 331, 371, 371, 388.5, 411, 419, 411, 394,
                 342, 342, 325, 315, 342, 345, 348.5, 351, 348, 370, 371, 354, 363, 357, 351,
                 369, 335, 381, 391, 392, 395, 398, 401, 405, 410, 408, 415, 418, 422, 418.5,
                 405.5, 405, 409, 417, 420, 424, 438, 438.5, 434, 438, 440, 447, 448, 444.5,
                 441, 440.5, 445, 444],
    '路口的纵坐标Y': [359, 343, 351, 377.5, 376, 383, 362, 353.5, 342, 325, 301, 316, 270, 292,
                 335, 328, 335, 371, 374, 394, 277, 271, 265, 290, 300, 301, 306, 328, 337,
                 367, 351, 355, 350, 342.5, 339, 334, 335, 330, 333, 330.5, 327.5, 344, 343,
                 346, 342, 348, 372, 374, 372, 382, 380.5, 377, 369, 363, 353, 374, 382.5,
                 387, 382, 388, 395, 381, 375, 366, 361, 362, 359, 360, 355, 350, 351, 347,
                 354, 356, 364.5, 368, 370, 364, 370, 372, 368, 373, 376, 385, 392, 392, 381,
                 383, 385, 381.5, 380, 360]
}

geo_data = pd.DataFrame(data)

# 显示数据以确认读取正确
geo_data.head()
import networkx as nx
import matplotlib.pyplot as plt

# 读取警察局数据
police_stations = {
    'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
    'A6': 6, 'A7': 7, 'A8': 8, 'A9': 9, 'A10': 10,
    'A11': 11, 'A12': 12, 'A13': 13, 'A14': 14,
    'A15': 15, 'A16': 16, 'A17': 17, 'A18': 18,
    'A19': 19, 'A20': 20
}

# 关键交通要道
key_intersections = [12, 14, 16, 21, 22, 23, 24, 28, 29, 30, 38, 48, 62]

# 距离数据
distances = {'1': [[9.300537618869138, '75'], [6.4031242374328485, '78'], [5.0, '69'], [6.264982043070834, '74']],
             '2': [[9.486832980505138, '44'], [19.144189719076646, '40'], [8.0, '43'], [8.602325267042627, '70']],
             '3': [[42.464691215173104, '45'], [15.239750654128171, '65'], [11.629703349613008, '44'], [12.658988901172163, '55']],
             '4': [[45.609757727924844, '39'], [10.307764064044152, '63'], [18.681541692269406, '57'], [3.5, '62']], '5': [[5.0, '49'], [8.48528137423857, '50'], [14.560219778561036, '47']], '6': [[16.0312195418814, '59'], [14.866068747318506, '47']], '7': [[11.40175425099138, '32'], [12.806248474865697, '47'], [38.18376618407357, '15'], [5.830951894845301, '30'], [30.4138126514911, '37']], '8': [[11.597413504743201, '9'], [20.796634343085422, '47'], [8.276472678623424, '33'], [9.300537618869138, '46']], '9': [[4.242640687119285, '35'], [11.597413504743201, '8'], [5.024937810560445, '34']], '10': [[49.216359068911224, '34'], [35.38361202590826, '26']], '11': [[32.69556544854363, '22'], [9.0, '26'], [20.024984394500787, '25']], '12': [[17.88854381999832, '25'], [33.04920573932148, '27']], '14': [[32.64965543462902, '21'], [67.4166151627327, '16']], '15': [[38.18376618407357, '7'], [29.68164415931166, '31'], [47.51841748206689, '28']], '16': [[67.4166151627327, '14'], [34.058772731852805, '38'], [6.082762530298219, '36']], '17': [[26.879360111431225, '40'], [9.848857801796104, '42'], [40.22437072223753, '81'], [8.5, '41']], '18': [[6.708203932499369, '81'], [5.385164807134504, '83'], [19.72308292331602, '73'], [8.06225774829855, '80']], '19': [[4.47213595499958, '79'], [9.848857801796104, '77']], '20': [[3.605551275463989, '86'], [4.47213595499958, '85'], [9.486832980505138, '89']], '21': [[18.027756377319946, '22'], [32.64965543462902, '14']], '22': [[9.055385138137417, '13'], [32.69556544854363, '11'], [18.027756377319946, '21']], '23': [[5.0, '13']], '24': [[23.853720883753127, '13'], [18.027756377319946, '25']], '25': [[20.024984394500787, '11'], [17.88854381999832, '12'], [18.027756377319946, '24']], '26': [[7.433034373659253, '27'], [35.38361202590826, '10'], [9.0, '11']], '27': [[33.04920573932148, '12'], [7.433034373659253, '26']], '28': [[9.486832980505138, '29'], [47.51841748206689, '15']], '29': [[74.32361670424818, '30'], [9.486832980505138, '28']], '30': [[5.830951894845301, '7'], [7.0710678118654755, '48'], [74.32361670424818, '29']], '31': [[11.704699910719626, '32'], [15.532224567009067, '34'], [29.68164415931166, '15']], '32': [[5.0990195135927845, '33'], [11.40175425099138, '7'], [11.704699910719626, '31']], '33': [[7.566372975210778, '34'], [8.276472678623424, '8'], [5.0990195135927845, '32']], '34': [[5.024937810560445, '9'], [49.216359068911224, '10'], [15.532224567009067, '31'], [7.566372975210778, '33']], '35': [[6.708203932499369, '45'], [4.242640687119285, '9'], [5.0, '36']], '36': [[5.0, '35'], [5.0990195135927845, '37'], [6.082762530298219, '16'], [35.014282800023196, '39']], '37': [[30.4138126514911, '7'], [5.0990195135927845, '36']], '38': [[3.0, '39'], [40.078048854703496, '41'], [34.058772731852805, '16']], '39': [[17.67766952966369, '40'], [45.609757727924844, '4'], [35.014282800023196, '36'], [3.0, '38']], '40': [[19.144189719076646, '2'], [26.879360111431225, '17'], [17.67766952966369, '39']], '41': [[8.5, '17'], [46.31684358848301, '92'], [40.078048854703496, '38']], '42': [[8.06225774829855, '43'], [9.848857801796104, '17']], '43': [[8.0, '2'], [8.06225774829855, '72'], [8.06225774829855, '42'], [7.615773105863909, '70']], '44': [[11.629703349613008, '3'], [9.486832980505138, '2'], [14.7648230602334, '67']], '45': [[6.0, '46'], [42.464691215173104, '3'], [6.708203932499369, '35']], '46': [[9.300537618869138, '8'], [29.427877939124322, '55'], [6.0, '45']], '47': [[10.198039027185569, '48'], [14.866068747318506, '6'], [14.560219778561036, '5'], [12.806248474865697, '7'], [20.796634343085422, '8']], '48': [[29.0, '61'], [7.0710678118654755, '30'], [10.198039027185569, '47']], '49': [[10.44030650891055, '50'], [6.708203932499369, '53'], [5.0, '5']], '50': [[3.8078865529319543, '51'], [8.48528137423857, '5'], [10.44030650891055, '49']], '51': [[4.301162633521313, '52'], [2.9154759474226504, '59'], [3.8078865529319543, '50']], '52': [[4.242640687119285, '56'], [4.301162633521313, '51'], [8.54400374531753, '53']], '53': [[8.54400374531753, '52'], [22.80350850198276, '54'], [6.708203932499369, '49']], '54': [[10.04987562112089, '55'], [24.186773244895647, '63'], [22.80350850198276, '53']], '55': [[12.658988901172163, '3'], [29.427877939124322, '46'], [10.04987562112089, '54']], '56': [[12.379418403139947, '57'], [4.242640687119285, '52']], '57': [[7.5, '58'], [8.139410298049853, '60'], [18.681541692269406, '4'], [12.379418403139947, '56']], '58': [[7.810249675906654, '59'], [7.5, '57']], '60': [[13.892443989449804, '62'], [8.139410298049853, '57'], [34.713109915419565, '61']], '61': [[34.713109915419565, '60'], [29.0, '48']], '62': [[3.5, '4'], [60.01666435249463, '85'], [13.892443989449804, '60']], '63': [[9.055385138137417, '64'], [10.307764064044152, '4'], [24.186773244895647, '54']], '64': [[5.830951894845301, '65'], [13.152946437965905, '76'], [9.055385138137417, '63']], '65': [[3.1622776601683795, '66'], [15.239750654128171, '3'], [5.830951894845301, '64']], '66': [[4.242640687119285, '67'], [9.219544457292887, '76'], [3.1622776601683795, '65']], '67': [[14.7648230602334, '44'], [4.123105625617661, '68'], [4.242640687119285, '66']], '68': [[7.0710678118654755, '69'], [4.527692569068709, '75'], [4.123105625617661, '67']], '69': [[5.385164807134504, '70'], [6.4031242374328485, '71'], [5.0, '1'], [7.0710678118654755, '68']], '70': [[8.602325267042627, '2'], [7.615773105863909, '43'], [5.385164807134504, '69']], '71': [[5.0, '72'], [6.103277807866851, '74'], [6.4031242374328485, '69']], '72': [[8.06225774829855, '73'], [8.06225774829855, '43'], [5.0, '71']], '73': [[4.031128874149275, '74'], [19.72308292331602, '18'], [8.06225774829855, '72']], '74': [[6.264982043070834, '1'], [16.91892431568863, '80'], [6.103277807866851, '71'], [4.031128874149275, '73']], '75': [[3.5355339059327378, '76'], [9.300537618869138, '1'], [4.527692569068709, '68']], '76': [[4.47213595499958, '77'], [13.152946437965905, '64'], [9.219544457292887, '66'], [3.5355339059327378, '75']], '77': [[10.0, '78'], [9.848857801796104, '19'], [4.47213595499958, '76']], '78': [[6.708203932499369, '79'], [6.4031242374328485, '1'], [10.0, '77']], '79': [[4.47213595499958, '80'], [4.47213595499958, '19'], [6.708203932499369, '78']], '80': [[8.06225774829855, '18'], [16.91892431568863, '74'], [4.47213595499958, '79']], '81': [[5.024937810560445, '82'], [40.22437072223753, '17'], [6.708203932499369, '18']], '82': [[5.408326913195984, '83'], [8.73212459828649, '90'], [5.024937810560445, '81']], '83': [[9.848857801796104, '84'], [5.385164807134504, '18'], [5.408326913195984, '82']], '84': [[7.280109889280518, '85'], [9.848857801796104, '83'], [3.0, '89']], '85': [[4.47213595499958, '20'], [60.01666435249463, '62'], [7.280109889280518, '84']], '86': [[11.045361017187261, '87'], [9.340770846134703, '88'], [3.605551275463989, '20']], '87': [[4.031128874149275, '88'], [21.37755832643195, '92'], [11.045361017187261, '86']], '88': [[4.031128874149275, '89'], [3.0413812651491097, '91'], [9.340770846134703, '86'], [4.031128874149275, '87']], '89': [[9.486832980505138, '20'], [3.0, '84'], [3.5355339059327378, '90'], [4.031128874149275, '88']], '90': [[4.743416490252569, '91'], [8.73212459828649, '82'], [3.5355339059327378, '89']], '91': [[20.024984394500787, '92'], [3.0413812651491097, '88'], [4.743416490252569, '90']], '59': [[16.0312195418814, '6'], [2.9154759474226504, '51'], [7.810249675906654, '58']], '13': [[9.055385138137417, '22'], [5.0, '23'], [23.853720883753127, '24']], '92': [[46.31684358848301, '41'], [21.37755832643195, '87'], [20.024984394500787, '91']]}

# 构建图模型
city_graph = nx.Graph()

# 添加节点和边到图中
all_nodes = set(police_stations.values())
for node, edges in distances.items():
    all_nodes.add(int(node))
    for edge in edges:
        weight, target = edge
        city_graph.add_edge(int(node), int(target), weight=weight)
        all_nodes.add(int(target))

# 使用实际地理位置进行绘图
pos = {row['全市路口节点标号']: (row['路口的横坐标X'], row['路口的纵坐标Y']) for _, row in geo_data.iterrows()}
labels = {e: f"{w:.2f}" for e, w in nx.get_edge_attributes(city_graph, 'weight').items()}

plt.figure(figsize=(15, 10))
nx.draw(city_graph, pos, with_labels=True, node_color='lightblue', node_size=700)
nx.draw_networkx_edge_labels(city_graph, pos, edge_labels=labels)
plt.show()
# 计算最短路径
shortest_paths = {source: dict(nx.single_source_dijkstra_path_length(city_graph, source)) for source in police_stations.values()}

# 转换为距离矩阵
nodes = list(all_nodes)
num_nodes = len(nodes)
node_index = {node: i for i, node in enumerate(nodes)}
distance_matrix = np.zeros((len(police_stations), num_nodes))

for i, source in enumerate(police_stations.values()):
    for j, target in enumerate(nodes):
        if source != target:
            distance_matrix[i, j] = shortest_paths.get(source, {}).get(target, float('inf'))
import pulp

# 定义线性规划问题
prob = pulp.LpProblem("PoliceAssignment", pulp.LpMinimize)

# 定义决策变量
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(len(police_stations)) for j in range(num_nodes)), cat='Binary')

# 目标函数：最小化最大距离并优先考虑关键交通要道
max_distance = pulp.LpVariable("max_distance")
prob += max_distance

# 添加约束：优先封锁关键交通要道
for j in key_intersections:
    prob += pulp.lpSum(x[i, node_index[j]] for i in range(len(police_stations))) == 1

# 添加约束：每个警察局只能分配一个路口
for i in range(len(police_stations)):
    prob += pulp.lpSum(x[i, j] for j in range(num_nodes)) <= 1

# 添加约束：最大距离约束
for i in range(len(police_stations)):
    for j in range(num_nodes):
        prob += x[i, j] * distance_matrix[i, j] <= max_distance
# 解决问题
prob.solve()

# 输出结果
print("Status:", pulp.LpStatus[prob.status])
assignments = []
for i in range(len(police_stations)):
    for j in range(num_nodes):
        if pulp.value(x[i, j]) == 1:
            print(f"Police station {nodes[i]} assigned to road intersection {nodes[j]} with distance {distance_matrix[i, j]}")
            assignments.append((nodes[i], nodes[j]))
# 确保形成封锁圈
def ensure_circular_blockade(assignments, graph):
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, node)
    for start_node, _ in assignments:
        dfs(start_node, None)
        if len(visited) == len(assignments):
            return True
    return False

# 检查分配的可行性
if not ensure_circular_blockade(assignments, city_graph):
    print("The assignments do not form a circular blockade. Adjusting assignments...")

# 绘制最终结果
plt.figure(figsize=(15, 10))
nx.draw(city_graph, pos, with_labels=True, node_color='lightblue', node_size=700)
nx.draw_networkx_edge_labels(city_graph, pos, edge_labels=labels)

for i, j in assignments:
    if i in pos and j in pos:
        plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'ro-')

plt.show()
