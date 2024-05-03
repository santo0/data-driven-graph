from dataclasses import dataclass
from typing import List, Tuple
from math import radians, cos, sin, asin, sqrt
import contextily as ctx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

NODE_LOCATION_FILE = './Node-Location.csv'
MATRIX_FILE = './data_matrix.csv'


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    r = 6371
    return c * r


@dataclass
class Node:
    row_id: int
    name: str
    eoi: str
    lat: float
    lon: float

    # no haversine formula
    @staticmethod
    def deprecated_distance(u: 'Node', v: 'Node') -> float:
        return ((u.lat-v.lat)**2+(u.lon-v.lon)**2)**(1/2)

    # haversine formula
    @staticmethod
    def distance(u: 'Node', v: 'Node') -> float:
        return haversine(u.lon, u.lat, v.lon, v.lat)


def deprecated_draw_graph(nodes: List[Node], edges: List[Tuple[str, str]]):
    G = nx.Graph()
    # draw nodes
    for node in nodes:
        G.add_node(node.name)
    # draw edges
    for u, v in edges:
        G.add_edge(u, v)

    # Draw the graph
    nx.draw(G, with_labels=True, node_color='skyblue',
            font_color='black', font_weight='bold')

    # Display the graph
    plt.show()


def get_graph(adj_matrix, name_mapping):
    # relabel_nodes(G, mapping, copy=True)
    G = nx.from_numpy_array(adj_matrix)
    nx.relabel_nodes(G, name_mapping, copy=False)
    return G


def draw_graph(G):
    # Define edge colors based on weights
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]

    # Plot the graph with edge colors
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color="skyblue",
            edge_color=edge_colors, width=2.0, edge_cmap=plt.cm.Blues)
    plt.show()


def df_to_gdf(df):
    geometry = [Point(lon, lat) for lon, lat in zip(df['Lon'], df['Lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def get_edges_from_adjacency_matrix(adj_matrix):
    edges = set()
    for i, row in enumerate(adj_matrix):
        for j, v in enumerate(row):
            if v != 0:
                edges.add(frozenset((i, j)))
    return [(i, j) for i, j in edges]


def draw_graph_in_map(gdf, edges: List[Tuple[int, int]], fig_name):
    # Plot the GeoDataFrame
    ax = gdf.plot(figsize=(10, 10), markersize=100, edgecolor='k')

    # Add basemap from contextily
    ctx.add_basemap(ax, crs=gdf.crs)
    # Plot edges between the nodes
    for i, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        for j, row2 in gdf.iterrows():
            edge = (i, j)
            if edge in edges:
                x2, y2 = row2.geometry.x, row2.geometry.y
                # Plotting edges with black color and transparency
                plt.plot([x, x2], [y, y2], 'r--',)
    ax.set_title(fig_name)

    plt.savefig(f'./fig/{fig_name}.png')


def run():
    df_loc = pd.read_csv(NODE_LOCATION_FILE, sep=';')
    for i, row in df_loc.iterrows():
        node = Node(row['Name'], row['EOI'], row['Lat'], row['Lon'], i)
        print(node.__dict__)


if __name__ == '__main__':
    run()
