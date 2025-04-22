import pandas as pd
import numpy as np

from kneed import KneeLocator


def find_elbow(distances,cv='convex'):
    
    lst_elbow = []
    
    for s in range(1,14):

        kneedle = KneeLocator(np.arange(distances.size),distances, S=s, curve=cv, direction='increasing')

#         print(kneedle.elbow_y,kneedle.knee_y)

        if pd.isnull(kneedle.elbow_y) and pd.isnull(kneedle.knee_y):
            continue

        lst_elbow.append(kneedle.elbow_y)
        lst_elbow.append(kneedle.knee_y)

    lst_elbow = np.unique(lst_elbow).tolist()

    return lst_elbow
