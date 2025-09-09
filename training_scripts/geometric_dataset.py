import torch 
import torch_geometric
from torch_geometric.data import Dataset, Data 

class SpatioTemporalDataset(Dataset):
    """
    A customized torch geometric dataset for spatiotemporal data,
    using the sliding window technique
    
    Args:
        data_array(np.ndarray): 3D data in form (nodes, features, timestamps)
        edge_index (np.ndarray): 2D np.array consists of edges in graph in form of (source, destination)
        edge_weight (np.ndarray): weight matrix with correspond to edges
        lookback (int): the length of past window used to predict the next days
        horizon (int): number of next days needs to be predcted
    """
    
    def __init__(self, data_array, edge_index, edge_attr=None, edge_weight=None, transform=None, lookback=30, horizon=1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.transform = transform
        
        self.x_data = torch.from_numpy(data_array).float()
        self.edge_index = torch.from_numpy(edge_index).long()
        
        
        self.edge_weight = None 
        self.edge_attr = None
        
        if (edge_attr is not None):
            self.edge_attr = torch.from_numpy(edge_attr).float()
        if (edge_weight is not None):
            self.edge_weight = torch.from_numpy(edge_weight).float()
        
        self._num_timestamps = data_array.shape[2]
    
    def __len__(self):
        """
        Return the length of available dataset depending on the length of lookback 
        and future horizon
        """
        return self._num_timestamps - self.lookback - self.horizon + 1
    
    def __getitem__(self, idx):
        start_x = idx 
        end_x = start_x + self.lookback
        
        start_y = end_x
        end_y = start_y + self.horizon
        
        x_window = self.x_data[:, :, start_x:end_x]
        y_window = self.x_data[:, :, start_y:end_y]  
        
        data = Data(
            x = x_window,
            edge_index = self.edge_index,
            edge_attr = self.edge_attr,
            edge_weight=self.edge_weight, 
            y=y_window
        )   
        
        if self.transform:
            data = self.transform(data)
        
        return data