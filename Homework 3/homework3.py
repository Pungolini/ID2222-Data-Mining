# %%
import collections
import random
import time
import argparse

DELETION = 'D'
ADDITION = 'A'
D = 0
U = 1
UV = 2


class Reservoir:

    def __init__(self, maxlen, undirected=True):

        self.G = collections.defaultdict(lambda: set())

        self.edges = list()
        self.edges_ = collections.defaultdict(lambda: tuple())
        self.M = maxlen
        self.undirected = undirected
        self.tau_local = collections.defaultdict(lambda: 0)
        self.tau = 0

    def add_edges(self, u, v):

        self.G[u].add(v)
        if self.undirected:
            self.G[v].add(u)

        self.edges.append((u, v))

    # self.edges_.update({(u,v) : (u,v)})

    def remove_edge(self, u, v):

        if u in self.G:
            if v in self.G[u]:
                self.G[u].remove(v)

        if self.undirected:
            if v in self.G:
                if u in self.G[v]:
                    self.G[v].remove(u)

        self.edges.remove((u, v))

    # del self.edges_[(u,v)]

    def edge_exists(self, u, v):
        return u in self.G and v in self.G[u]

    def get_shared_neighbors(self, u, v):
        return self.G[u] & self.G[v]

    def update_counters(self, operation, u, v, t, improved=False):

        if improved:
            eta = max(1, (t - 1) * (t - 2) / (self.M * (self.M - 1)))
        else:
            eta = 1

        inc = -eta if operation == DELETION else eta
        inc = int(inc)

        for c in self.get_shared_neighbors(u, v):

            self.tau += inc
            self.tau_local[c] += inc
            self.tau_local[u] += inc
            self.tau_local[v] += inc

            # Delete edges with counters = 0

            if operation == DELETION:
                for x in [c, u, v]:
                    if self.tau_local[x] <= 0:
                        self.tau_local.pop(x, None)  # Remove

    def biased_coin_flip(self, p):
        return True if random.random() <= p else False

    def sample_edge_base(self, t, improved=False):

        assert t > 0

        if t <= self.M:
            return True

        elif self.biased_coin_flip(self.M / t):

            # Randomly choose an edge
            del_idx = random.randint(0, len(self.edges) - 1)

            z, w = self.edges[del_idx]

            self.remove_edge(z, w)

            if improved == False:  # In improved version the counters should not be updated
                self.update_counters(DELETION, z, w, t, improved)

            return True
        return False


def init_stream(filename):
    with open('data/' + filename, 'rb') as f:
        for line in f:
            if line[0] != '#':  # Skip comments
                u, v = line.rstrip().split()[:2]
                if u != v:
                    yield u, v


def TRIEST(file, M=10000, improved=False, debug=True):
    S = Reservoir(M)

    # Initialize stream and read chunck by chunk
    stream = init_stream(file)

    t = 0
    op = '+'
    x = M // 10

    # Get stream data
    while stream:
        try:
            u, v = next(stream)

            if not S.edge_exists(u, v):
                t += 1
                if improved == True:
                    S.update_counters(ADDITION, u, v, t, True)
                # Sample edge using reservoir sampling
                if S.sample_edge_base(t, improved):
                    S.add_edges(u, v)
                    if improved == False:
                        S.update_counters(ADDITION, u, v, t, False)

                if debug and (t % x == 0):
                    print(f"t = {t}  Triangles: {int(S.tau)}\n")

        except StopIteration:
            break
    return S.tau


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Argument Parser. Please choose the graph dataset, the available memory'
                    'and the TRIEST variant')

    parser.add_argument('--dataset', default='grqc.txt', choices=['grqc.txt', 'caida.txt', 'stanford.txt'])
    parser.add_argument('--variant', default='base', choices=['base', 'improved'])
    parser.add_argument('--debug', default="n", choices=["y", "n"])
    parser.add_argument('--memory', default=15000, type=int)

    args = parser.parse_args()
    improved = True if args.variant == 'improved' else False
    debug_flag = False if args.debug == 'n' else True
    graph_file = args.dataset
    memory = args.memory

    # Execute the algorithm
    if memory <= 6:
        print("Try again with memory > 6")
        exit(0)

    print(f"Running Triest-{args.variant} on dataset  {args.dataset} with memory = {memory}")

    start = time.time()

    triangles = TRIEST(graph_file, M=memory, improved=improved, debug=debug_flag)

    end = time.time()

    print(f"M =  {memory} \tNumber of triangles found: {triangles}\n\n")
    print(f"Execution time: {end - start:.2f} seconds\n")
