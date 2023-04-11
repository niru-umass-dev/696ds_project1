import matplotlib.pyplot as plt
from statistics import stdev
from scipy.stats import spearmanr
import numpy as np

def get_centtend_measures(x, y):
    x = [100*value for value in x]
    y = [100*value for value in y]
    plt.scatter(x, y)
    return np.mean(y), np.std(y), spearmanr(x,y).statistic, np.corrcoef(x,y)[0,1], plt