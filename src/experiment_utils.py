import matplotlib.pyplot as plt
from statistics import stdev
from scipy.stats import spearmanr
import numpy as np

def get_centtend_measures(x, y, metric: str = None, label: str = None):
    x, y = np.asarray(x), np.asarray(y)
    if metric == 'ds':
        x,y = x,y
    elif metric == 'bs':
        x = _get_complement(x, src=(1, 0), tgt = (0, 1))
        y = _get_complement(y, src=(1, 0), tgt = (0, 1))
        # print()
        # print("resultant arrays")
        # print("x", x)
        # print("y", y)
    elif metric == 'nli' and label == 'agg':
        x, y = (x+1)/2, (y+1)/2
        # print()
        # print("resultant arrays")
        # print("x", x)
        # print("y", y)
    else:
        # print("else condition")
        x,y = x,y
    x, y = x.tolist(), y.tolist()
    x = [100*value for value in x]
    y = [100*value for value in y]
    plt.scatter(x, y)
    return np.mean(y), np.std(y), spearmanr(x,y).statistic, np.corrcoef(x,y)[0,1], plt

def _get_complement(data, src=(1, 0), tgt = (0, 1)):
    data = (1 - (data/np.absolute(src[1] - src[0]))) * (tgt[1] - tgt[0])
    return data


if __name__ == '__main__':
    # d1 = -np.random.rand(48)
    # d2 = -np.random.rand(48)
    
    d1 = np.ones(48)
    d2 = np.zeros(48)
    print()
    print("original arrays")
    print("d1", d1)
    print("d2", d2)
    results = get_centtend_measures(d1, d2, 'bs')
    pass