import matplotlib.pyplot as plt
import numpy as np
from homework3 import *

"""
Generates plots for given datasets
"""

if __name__ == '__main__':

    triangles_dict = dict()
    runtime = dict()

    files = {0:"grqc.txt",1:"caida.txt",2:"stanford.txt"}
    REAL_TRIANGLE_COUNT = {0:48260,	1:36365,2:11329473}
    NUMBER_OF_EDGES = {0:14496 , 1:	106762,2:2312497}
    VARIANT = "impr"
    FILE = 1 # Set 0 for file1 and 1 for file2 and 2 for file3

    impr = True if VARIANT == "impr" else False

    min_mem = NUMBER_OF_EDGES[FILE]//5
    max_mem = NUMBER_OF_EDGES[FILE] + min_mem # Do not try with the stanford dataset
    for memory in np.linspace(min_mem,max_mem,15,dtype=int):

        start = time.time()

        triangles = TRIEST(files[FILE],M = memory,improved = impr,debug = False)

        end = time.time()
        triangles_dict[memory] = triangles
        runtime[memory] = round(end - start,2)

        print(f"Memory: {memory}\tTriangles: {triangles}\tRuntime: {runtime[memory]} s")


    plt.plot(list(triangles_dict.keys()), list(triangles_dict.values()),marker='.')
    plt.title(f"Triangle count vs Memory for dataset {files[FILE]}")
    plt.xlabel("Memory")
    plt.ylabel("Triangle count")
    plt.axhline(y=REAL_TRIANGLE_COUNT[FILE], color='y', linestyle='--',label = "Real triangle count")
    plt.axvline(x = NUMBER_OF_EDGES[FILE],color = 'g', linestyle='--', label = "Number of edges")
    plt.legend(loc = "center right")
    plt.show()

    plt.plot(list(runtime.keys()), list(runtime.values()),marker = '.')
    plt.title(f"Runtime vs Memory for dataset {files[FILE]}")
    plt.xlabel("Memory")
    plt.ylabel(f"TRIEST-{VARIANT} Runtime [s]")
    plt.show()

    variant = "impr" if impr else "base"
    with open(f"output/triest-{variant}-"+files[FILE], "w") as outfile:

        for m,tri,t in zip(triangles_dict.keys(),triangles_dict.values(),runtime.values()):

            outfile.write(f"{m};{t}\n")






