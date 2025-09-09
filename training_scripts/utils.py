import pandas as pd 
import numpy as np


def convert3dtensor(nodes, df: pd.DataFrame):
    """ 
    Convert 2D sp500 dataframe into 3D shape tensor 
    which is in form (n_nodes, n_timestamps, n_features)
    
    Args:

        - df (pd.DataFrame): Dataframe needed to be converted
        - nodes (np.ndarray): A np.ndarry consists of stock labels
    Return:
        (np.ndarray): Data returned in 3D shape
    """
    df.set_index(['Symbol', 'Date'], inplace=True)
    
    return np.stack(
        [df.loc[node].values.T for node in nodes],
        axis=0
    ).transpose(0, 2, 1)


def load_data(train=True):
    
    """ 
    A funtion aims to split inital data into train, test, split using Time-series K-fold split technique.
    
    The last month data is considered as the test dataset.
    
    Args:
        - train (bool): If True, return 4 folds, each fold consists of both training & validation sets. 
        Otherwise, it would return test set with entire training set.
    Return:
        (list): Set of couples, each couple consists of 2 sets, the second following the first by time
    """
    
    df = pd.read_csv('../input/clear_sp500.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    nodes = df['Symbol'].unique()
    
    timelines = [
        pd.Timestamp(year=2025, month=4, day=1, tz='UTC'),
        pd.Timestamp(year=2025, month=5, day=1, tz='UTC'),
        pd.Timestamp(year=2025, month=6, day=1, tz='UTC'),
        pd.Timestamp(year=2025, month=7, day=1, tz='UTC'),
        pd.Timestamp(year=2025, month=8, day=1, tz='UTC')
    ]
    
    predicted_range = pd.Timedelta(days=30)
    
    folds = []
    
    if train:
        
        for i in range(4):
            train_df = df[df['Date'] < timelines[i]].copy().reset_index(drop=True)
            valid_df = df[(df['Date'] >= timelines[i]) & (df['Date'] <= 
                        
                        timelines[i] + predicted_range)].copy().reset_index(drop=True)
            
            folds.append((convert3dtensor(nodes, train_df), 
                          convert3dtensor(nodes, valid_df)))
    
    else:
        train_df = df[df['Date'] < timelines[4]].copy().reset_index(drop=True)
        
        test_df = df[df['Date'] >= timelines[4]].copy().reset_index(drop=True)
        
        folds.append((convert3dtensor(nodes, train_df),
                     convert3dtensor(nodes, test_df)))
        
    return folds
        
def load_edge(adj_path="../input/adj_ae-bert.npy"):
    """ 
    A function aims to construct edge-related matrix using the 
    pre-defined adjacency matrix
    
    Args:
        adj_path (str): Path to the corresponding adjacency matrix, which is 
        either correlation adjacency matrix or AE combined BERT adjacency matrix
        
    Return:
        (np.ndarray, np.ndarray): 2 edge-related matricies, the first one
        is edge_index matrix, which is in form of (2, num of edges), each row representing for
        (source, destination), and the other is edge_weight matrix in form of (edge_weight,) each row
        representing for weight of the corresponding edge
    """
    adj_matrix = np.load(adj_path)
    
    nodes_nb = len(adj_matrix)
    edge_nb = np.count_nonzero(adj_matrix)
    edge_index = np.zeros((2, edge_nb))
    edge_weight = np.zeros((edge_nb))
    count = 0
    
    for i in range(nodes_nb):
        for j in range(nodes_nb):
            if (weight := adj_matrix[i, j]) != 0:
                edge_index[0, count], edge_index[1, count] = i, j
                edge_weight[count] = weight
                count += 1
                
    return edge_index, edge_weight