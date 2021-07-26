from sklearn.model_selection import StratifiedKFold
import pandas as pd


def kfold_splitter(df=df_train_val):
    from sklearn.model_selection import StratifiedKFold 
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
    train_idx = []
    val_idx = []
        
    for train_index, val_index in kfold.split(df.index, df['Year']):
        train_idx.append(L(train_index, use_list=True))
        val_idx.append(L(val_index, use_list=True))    
        
    return train_idx, val_idx

def get_fold(split_list, fold=0):
    def _inner(o):
        return split_list[0][fold], split_list[1][fold]
    return _inner