import networkx as nx
from scripts.energy_cost_model import predict_energy_cost, train_energy_cost_model

# Function to calculate the energy cost between two points
def energy_cost(model_positive, model_negative, df_altitude, x1, y1, x2, y2, bodymass, pointsresolution, walkingspeed):
    altitude_diff = df_altitude.at[y2, x2] - df_altitude.at[y1, x1]
    return predict_energy_cost(altitude_diff, model_positive, model_negative) * bodymass * (pointsresolution / walkingspeed)

def create_weighted_graph(model_positive, model_negative, df_altitude, bodymass, pointsresolution, walkingspeed):
    G = nx.DiGraph()
    for y in range(df_altitude.shape[0]):
        for x in range(df_altitude.shape[1]):
            G.add_node((x, y))

    for y in range(df_altitude.shape[0]):
        for x in range(df_altitude.shape[1]):
            if x + 1 < df_altitude.shape[1]:
                weight1 = energy_cost(model_positive, model_negative, df_altitude, x, y, x + 1, y, bodymass, pointsresolution, walkingspeed)
                weight2 = energy_cost(model_positive, model_negative, df_altitude, x + 1, y, x, y, bodymass, pointsresolution, walkingspeed)
                G.add_edge((x, y), (x + 1, y), weight=weight1)
                G.add_edge((x + 1, y), (x, y), weight=weight2)
            if y + 1 < df_altitude.shape[0]:
                weight1 = energy_cost(model_positive, model_negative, df_altitude, x, y, x, y + 1, bodymass, pointsresolution, walkingspeed)
                weight2 = energy_cost(model_positive, model_negative, df_altitude, x, y + 1, x, y, bodymass, pointsresolution, walkingspeed)
                G.add_edge((x, y), (x, y + 1), weight=weight1)
                G.add_edge((x, y + 1), (x, y), weight=weight2)

    return G

def find_shortest_path(model_positive, model_negative, G, target, df_altitude, bodymass, pointsresolution, walkingspeed):
    H = G.reverse()
    shortest_path_lengths = nx.single_source_dijkstra_path_length(H, source=target, weight='weight')
    y_coordinate_line = 0

    shortest_paths_along_line = {}
    for node, distance in shortest_path_lengths.items():
        x, y = node
        if y == y_coordinate_line:
            shortest_paths_along_line[node] = distance

    if shortest_paths_along_line:
        source = min(shortest_paths_along_line, key=shortest_paths_along_line.get)
        shortest_path = nx.shortest_path(H, source=source, target=target, weight='weight')
        shortest_path.reverse()
        total_energy = calculate_total_energy_cost(model_positive, model_negative, shortest_path[::-1], df_altitude, bodymass, pointsresolution, walkingspeed)

        return shortest_path, total_energy
    else:
        return None, None

def calculate_total_energy_cost(model_positive, model_negative, path, df_altitude, bodymass, pointsresolution, walkingspeed):
    total_energy_cost = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        altitude_diff = df_altitude.at[y2, x2] - df_altitude.at[y1, x1]
        energy_cost = predict_energy_cost(altitude_diff, model_positive, model_negative) * bodymass * (pointsresolution / walkingspeed)
        total_energy_cost += energy_cost
    return total_energy_cost
