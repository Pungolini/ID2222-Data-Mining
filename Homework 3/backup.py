#%%
import collections
import random
import time

DELETION = 'D'
ADDITION = 'A'
D = 0
U = 1
UV = 2

class Reservoir:

    def __init__(self,maxlen,undirected=True):

        self.G = collections.defaultdict(lambda:set())

        self.edges = list()
        self.edges_ = dict()
        self.M = maxlen
        self.undirected = undirected
        self.tau_local = collections.defaultdict(lambda :0)
        self.tau = 0
        
    def add_edges(self,u,v):

        self.G[u].add(v)
        if self.undirected:
            self.G[v].add(u)
        
        self.edges.append((u,v))
        #self.edges_[(u,v)] = (u,v)
    
    def remove_edge(self,u,v):
        
        if u in self.G:
            if v in self.G[u]:
                self.G[u].remove(v)

        if self.undirected:
            if v in self.G:
                if u in self.G[v]:
                    self.G[v].remove(u)

        self.edges.remove((u,v))
        #del self.edges_[(u,v)]
    
    def edge_exists(self,u,v):
        return (u,v) in self.edges or (v,u) in self.edges

    def get_shared_neighbors(self,u,v):
        return self.G[u] & self.G[v]


    def update_counters(self,operation,u,v,t,improved = False):

        eta = max(1, (t-1)*(t-2)/(self.M*(self.M-1)) ) if improved else 1

        inc =  -eta if operation == DELETION else eta
        
        for c in self.get_shared_neighbors(u,v):

            self.tau += inc
            self.tau_local[c] += inc
            self.tau_local[u] += inc
            self.tau_local[v] += inc

            # Delete edges with counters = 0
            
            if operation == DELETION:
                for x in [c,u,v]:
                    if self.tau_local[x] <= 0:
                        self.tau_local.pop(x,None) # Remove

    def biased_coin_flip(self,p):
        return True if random.random() < p else False
    
    def sample_edge_base(self,t,improved = False):

        assert t > 0

        if t <= self.M:
            return True
        
        elif self.biased_coin_flip(self.M/t):

            # Randomly choose an edge
            # TODO: keeping the self.edges as a list has noticeable performance decrease
            # TODO: Implement self.edges as a dict
            del_idx = random.randint(0,len(self.edges)-1)

            u_,v_ = self.edges[del_idx]

            self.remove_edge(u_,v_)

            if improved == False: # In improved version the counters should not be updated
                self.update_counters(DELETION,u_,v_)

            return True
        return False


    def sample_edge_del(self,t,u,v,d0,di):

        assert t > 0

        if d0 + di == 0:

            if len(self.edges) <= self.M:
                self.add_edges(u,v)
                return True, d0,di
            
            elif self.biased_coin_flip(self.M/t):

                # Randomly choose an edge
                # TODO: keeping the self.edges as a list has noticeable performance decrease
                # TODO: Implement self.edges as a dict
                del_idx = random.randint(0,len(self.edges)-1)

                z,w = self.edges[del_idx]

                # Update the counters
                self.update_counters(DELETION,z,w)

                self.remove_edge(z,w)
                self.add_edges(u,v)
                return True, d0,di
        elif self.biased_coin_flip(di/(d0+di)):
            self.add_edges(u,v)
            di -= 1
            return True , d0,di
        else:
            d0 -= 1
            return False ,d0,di

def init_stream(filename):

    with open('data/'+filename, 'r') as f:
        t = 0
        for line in f:
            if line[0] != '#': # Skip comments
                u,v = line.rstrip().split()[:2]
                if u != v:
                    yield u,v


def TRIEST_1(file,M = 10000,improved = False,debug = True):

    S = Reservoir(M)

    # Initialize stream and read chunck by chunk
    stream = init_stream(file)

    t = 0
    op = '+'
    x = M//10

    # Get stream data
    while stream:
        try:
            u,v = next(stream)
            if debug and (t % x == 0):
                print(f"t = {t}  Triangles: {S.tau}\n")
            
            if not S.edge_exists(u,v):
                t += 1
                if improved:
                    S.update_counters(ADDITION,u,v,t,improved)
                # Sample edge using reservoir sampling
                if S.sample_edge_base(t,improved):
                    S.add_edges(u,v)
                    if not improved:
                        S.update_counters(ADDITION,u,v,t,improved)

        except StopIteration:
            break
    return S.tau

def TRIEST_FD(file,M = 10000,debug = True):

    S = Reservoir(M)
    d0 = 0
    di = 0
    s = 0
    t = 0

    # Initialize stream and read chunck by chunk
    stream = init_stream(file)

    op = ADDITION
    inc = 1 if op == ADDITION else -1
    x = M//10

    # Get stream data
    while stream:
        try:
            u,v = next(stream)

            if debug and (t % x == 0):
                print(f"t = {t}  Triangles: {S.tau}\n")
            

            t += 1
            s = s + inc

            # Sample edge using reservoir sampling

            flag, d0,di = S.sample_edge_del(t,u,v,d0,di)
            if flag:
                S.update_counters(ADDITION,u,v)
            elif S.edge_exists(u,v):
                S.update_counters(DELETION,u,v)
                S.remove_edge(u,v)
                di += 1
            else:
                d0 += 1


        except StopIteration:
            break

    return S.tau

if __name__ == '__main__':
    import argparse


    files = ("CA-GrQc.txt","as-caida20071105.txt")
    parser = argparse.ArgumentParser(description='Argument Parser. Please choose the graph dataset, the available memory'
												 'and the TRIEST variant')

    parser.add_argument('--memory', default=15000, type=int)

    args = parser.parse_args()

    memory = args.memory

    print(f"Running Triest with memory = {memory} on dataset  ")

    start = time.time()

    triangles = TRIEST_1(files[1],M = memory,improved = False,debug = False)

    end = time.time()


    print(f"M =  {memory} \tNumber of triangles found: {triangles}\n\n")
    print(f"Execution time: {end - start:.2f} seconds\n")
