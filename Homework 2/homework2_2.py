# %%
import itertools as it
import time
from collections import defaultdict


def count_item(T, L, s):
    result = set()
    in_frequent = set()
    count_table = dict()
    for element in L:
        count_table[element] = 0
        for transaction in T:
            c = int(set(element).issubset(transaction))
            # print(f"Count of {element} in {transaction} = {c}")
            count_table[element] += c

            if count_table[element] >= s:
                result.add(element)
                break
        if count_table[element] < s:
            in_frequent.add(element)

    return result, in_frequent


def a_priori(T, s, k_max=3):
    L1 = set()
    count = defaultdict(int)
    for transaction in T:
        for element in transaction:
            count[element] +=1


    L1 = {item for item in count if count[item] >= s}


    print(f"len L1 = {len(L1)}   Total = {len(count)}")

    print("Finished creating L1")

    L = [L1] + [None for _ in range(k_max)]
    C = [None for _ in range(k_max + 1)]
    in_frequent = set()

    k: int = 1
    while len(L[k - 1]) > 1:
        # print(f"Progress: {(k/set_size)*100:.3f} %")
        print(f"len = {len(L[k - 1])}")
        # 1. generate k-sized itemsets
        if k > 1:
            print(f"L[k-1] = {L[k - 1]}")
            print("INFREQUENT: ", in_frequent)
            C[k] = gen_Ck(L[1], k, in_frequent)
            print("C: ", C[k])
        else:
            C[k] = L1

        # print(f"C{k} = {C[k]}")

        # 2.Count the frequency of each itemset
        L[k], in_frequent = count_item(T, C[k], s)
        print("L: ", L[k])

        # print(f"Count table: {count_table}")

        # L[k] = {element for element in count_table if count_table[element] >= s}

        # print(f"L{k} after = {L[k]}")
        k += 1

    return L[1:]


def gen_Ck(L_1, k, in_frequent):
    not_items = set()
    temp_items = set()
    if k < 3:
        items = set(it.combinations(L_1, k))
    else:
        temp = set(it.combinations(L_1, k))
        for i in in_frequent:
            for t in temp:
                if set(i).issubset(t):
                    not_items.add(t)
                else:
                    temp_items.add(t)
        items = temp_items-not_items
    return items


if __name__ == '__main__':

    begin = time.time()

    transactions_file = "T10I4D100K.dat"

    T_db = []
    unique_elements = set()

    N_BASKETS = 0

    with open(transactions_file, "r") as T_file:

        T_file = T_file.readlines()

        for line in T_file:
            N_BASKETS += 1
            int_set = tuple(x for x in line.rstrip().split(" "))
            unique_elements.add(int_set)
            T_db.append(int_set)

    T = [("A", "D", "C"), ("B", "C", "E"), ("A", "B", "C", "E"), ("B", "E")]

   # min_support = max(2, int(0.001 * N_BASKETS))
    min_support = 300

    print("Finished reading file")
    print("MIN_SUPPORT: ", min_support)

    pruned = a_priori(T_db, s=min_support)
    end = time.time()
    print("Runtime: ", end-begin)
    print(pruned)

# %%
