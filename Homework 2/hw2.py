#%%
import itertools as it
from collections import defaultdict
from time import time


def count_items(T:list,item_set:set,s:int):
    count_table = defaultdict(int)

    x = len(item_set)//10

    for i,item in enumerate(item_set):

        if i%x != 0:
            print(f" {i+1}/{len(item_set)}")

        count_table[item] = 0
        for transaction in T:
            transaction = transaction.split(' ')           
            c = int( set(item).issubset(set(transaction)) )
            count_table[item] += c
            if False:
                print(f"Count of {set(item)} in {set(transaction)} = {c} ")

            # No need to count until the end, min_support is verified , i think
            #if count_table[item] >= s:
            #    result.add(item)
            #    break

    #print(f"Count table = {count_table}")
    #print(f"Elements pruned = {item_set - result}")
    return count_table



def pruned(candidates,frequent_items):

    frequent_pairs = set()

    for pair in candidates:
        skip_pair = False
        subset = [x for x in pair]

        # For each subset check if it is in frequent set
        #print(f"Subset of {pair} is {subset}\n")
        
        for s in subset:
            #print(f"S^F = {s&frequent[k-1]} \n")
            #print(f"Testing if {s} is in {frequent_items} -> {s in frequent_items}")
            if s not in frequent_items:
                #print(f"{s} is not in {frequent_items}")
                skip_pair = True
                print(f"Skipped {pair}")
                break
        
        if skip_pair == False:
            frequent_pairs.add(pair)

    
    return frequent_pairs





if __name__ == '__main__':

    transactions_file = "T10I4D100K.dat"

    T_db = []
    N_BASKETS = 0
    frequent_items = set()
    counts = defaultdict(int)

    i = 0
    start = time()
    T = ["A D C", "B C E","A B C E","B E"]
    with open(transactions_file,"r") as T_file:
        T_db = T_file.readlines()
        for line in T_db:#T_file.readlines():
            N_BASKETS += 1
            for item in line.split(" "):
                counts[item] += 1
    


    print("Finished reading file")

    min_support = max(2,int(0.01*N_BASKETS))
    print(f"Minimum support threshold: {min_support}")

    # First pass
    frequent_items = {item for item in counts if counts[item] >= min_support}

    # Second pass
    counts_pair = defaultdict(int)

    candidates = list(it.combinations(frequent_items,2))

    #print(f"Candidates = { candidates}")

    frequent_pairs = pruned(candidates,frequent_items)

    print(f"len = {len(frequent_pairs)}")

    count_pairs = count_items(T_db,frequent_pairs,min_support)

    print(f"count = {counts_pair}")

    
    frequent_pairs = {key for key in count_pairs if count_pairs[key] >= min_support}

    print(f"k = 2 frequent pairs = {frequent_pairs}")

    end = time()


    print(f"Took {end-start:.3f} sec")



    #candidates =  list(it.combinations(frequent_items,3))

