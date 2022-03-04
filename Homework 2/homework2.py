#%%
import itertools as it
from collections import defaultdict

from pprint import pprint
def count_item(T:list,item_set:set,s:int):
    count_table = dict()
    result = set()

    x = len(item_set)//10

    for i,item in enumerate(item_set):

        count_table[item] = 0

        if i%x == 0:
            print(f"Item {i+1}/{len(item_set)}")

        for transaction in T:           
            c = int( set(item).issubset(transaction) )
            count_table[item] += c
            #print(f"Count of {item} in {transaction} = {c}")

            # No need to count until the end, min_support is verified , i think
            if count_table[item] >= s:
                result.add(item)
                break

        #if count_table[item] < s:
            #print(f"Element {item} needs to be pruned. Count = {count_table[item]}")

        #    item_set = item_set - set(item)

            #print(f"L after pruning: {L}")


    print(f"Count table = {count_table}")
    #print(f"Elements pruned = {item_set - result}")
    return result


def get_candidates(k,frequent_set):

    return list(it.combinations(frequent_set, k))



def prune_candidates(k,all_candidates,frequent_set):

    pruned_candidates = {}

    for pair in all_candidates:
        skip_pair = False
        
        # Generate all subsets of pair
        subset = [set(x) for x in gen_Ck(pair,k-1)]

        # For each subset check if it is in frequent set
        #print(f"Subset of {pair} is {subset}\n")
        
        for s in subset:
            #print(f"S^F = {s&frequent[k-1]} \n")
            #print(f"Testing if {s} is in {frequent[k-1]} -> {s == s&frequent[k-1]}")
            if s == s&frequent[k-1]:
                skip_pair = True
                #print(f"Skipped {pair}")
                break
        
        if not skip_pair:
            pruned_candidates.add(pair)






def a_priori(T,singletons,s,k_max = 3):

    L = singletons
    C = {1:L}

    k = 2

    counts = {i:None for i in range(k_max)}

    frequent = defaultdict(int)

    frequent[1] = singletons

    # First Pass count

    

    for k in range(2,k_max):

        all_candidates = get_candidates(k,frequent[k-1])

        # Generate all k-sized itemsets
        #all_candidates = gen_Ck(frequent[k-1],k) if k <=2 else singletons

        print(f"\n\nk ={k} Candidates: {list(all_candidates)}  len = {len(all_candidates)}")

        # Candidates are those pairs whose all subsets are in frequent set


        # Count support for elements in candidates
        #frequent.update({k:count_item(T,candidates,s)})
        # Candidates should be much smaller than combinations


        # Keep only candidates that verify support

    return L


def gen_Ck(item_set: set,k: int) -> set:
    """
    Returns a set of all k-sized combinations from item_set
    """

    #print(f"Generating {k}-combinations of {item_set}\n")

    """
    unique_elements = set()
    for item in item_set:
        unique_elements = unique_elements | set(item)
    """

    #unique_elements = set(set(item_A) | set(item_B) for item_A in item_set for item_B in item_set)

    #print(f"Unique elements: {unique_elements}")

    return list(it.combinations(item_set,k))

        

if __name__ == '__main__':

    transactions_file = "T10I4D100K.dat"

    T_db = []
    N_BASKETS = 0
    unique_elements = set()
    counts = defaultdict(int)

    i = 0
    T = ["A D C", "B C E","A B C E","B E"]
    with open(transactions_file,"r") as T_file:
        for line in T:#T_file.readlines():
            N_BASKETS += 1
            for item in line.split(" "):
                counts[item] += 1
    


    print("Finished reading file")

    min_support = max(2,int(0.01*N_BASKETS))
    print(f"Minimum support threshold: {min_support}")

    frequent_singletons = set(counts.keys())#{item for item in counts if counts[item] >= min_support}

    print(f"Number of unique_elements = {len(unique_elements)}\nNumber of total elements = {len(counts)}")

    print(f"singletons = {frequent_singletons}")
    pruned = a_priori(T_db,singletons=frequent_singletons,s=min_support)
    #print(pruned)

#%%