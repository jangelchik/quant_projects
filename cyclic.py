import numpy as np
import pandas as pd

def cyclic(col:pd.Series,
           max_val:float):
    
    sine = pd.Series([np.sin((2 * np.pi * i)/max_val) for i in list(col)],
                     name=f'{col.name}_sin',
                     index=col.index)
    cosine = pd.Series([np.cos((2 * np.pi * i)/max_val) for i in list(col)],
                 name=f'{col.name}_cos',
                 index=col.index)
    
    return pd.concat([sine,cosine],axis=1)
