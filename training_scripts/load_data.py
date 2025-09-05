import pandas as pd 
import numpy as np


def load_data(train=True):
    
    """ 
    A funtion aims to split inital data into train, test, split using Time-series K-fold split technique.
    
    The last month data is considered as the test dataset.
    
    Args:
        - train (bool): If True, return 4 folds, each fold consists of both training & validation sets. 
        Otherwise, it would return test set with entire training set.
    """
    
    df = pd.read_csv('../input/clear_sp500.csv', index_col=0)
    df['Date'] = pd.to_datetime(df['Date'])
    
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
            
            folds.append((train_df, valid_df))
    
    else:
        train_df = df[df['Date'] < timelines[4]].copy().reset_index(drop=True)
        
        test_df = df[df['Date'] >= timelines[4]].copy().reset_index(drop=True)
        
        folds.append((train_df, test_df))
        
    return folds
        
        