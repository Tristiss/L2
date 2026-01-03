import numpy as np
import random
import matplotlib.pyplot as plt
import time

from L2_Auswertung import *

def test_auswertung():
    x = np.linspace(0.6,6.2,100)
    y = [fit_func([1, 3.1, 0.2], i) + random.uniform(-0.1,0.1) for i in x]

    evaluation(x,y, 0)

    plt.errorbar(x,y, xerr = uncertainty_frequency, yerr = uncertainty_amplitude, label = "Test", capsize = 2)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

test_auswertung()

def test_random():
    x = np.linspace(0, 10, 100)
    data = [np.sin(i) + random.uniform(-0.2, 0.2) for i in x]

    maximum = max(data)

    plt.plot(x, data)
    plt.scatter(x[data.index(maximum)], maximum)
    plt.tight_layout()
    plt.grid()
    plt.show()


def test_lists():

    dicts:dict = {}
    lists:list = []

    times:list = []
    appendage:list = []
    n = 100000

    for i in range(n):
        li = [np.float64(random.uniform(0,2)) for j in range(1000)]
        start = time.perf_counter()
        dicts.update({i:li})
        end = time.perf_counter()
        appendage.append(i)
        times.append(end - start)

    plt.loglog(appendage, times)
    plt.tight_layout()
    plt.grid()
    plt.show()

def i_dont_remeber_what_this_was():
        times = np.linspace(0,100, 1000)
        data = times**2

        data_tmp = {"time":times, "amplitude":data}

        path = "C:"

        pd.DataFrame(data_tmp).to_csv(path + rf"\waves\Messung_.csv", sep = ";")

