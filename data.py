import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from graph import Node

NODE_LOCATION_FILE = './Node-Location.csv'
DATA_FILE = './data_matrix.csv'


def get_nodes():
    df_loc = pd.read_csv(NODE_LOCATION_FILE, sep=';')
    nodes = []
    node_ids = {}
    for i, row in df_loc.iterrows():
        node = Node(
            row_id=i, name=row['Name'], eoi=row['EOI'], lat=row['Lat'], lon=row['Lon'])
        nodes.append(node)
        node_ids[i] = node.name
    return nodes, node_ids


def get_scaled_data():
    df = pd.read_csv(DATA_FILE, sep=';')
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    # Extract numerical columns for scaling
    numerical_columns = df.columns[1:]
    # Scale the numerical columns using StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numerical_columns])

    # Replace the original values with scaled values in the DataFrame
    df[numerical_columns] = scaled_values
    return df

def get_data():
    df = pd.read_csv(DATA_FILE, sep=';')
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    return df
