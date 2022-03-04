#%%
import itertools as it

def count_item(T,L,s):
    result = set()
    garbage = set()
    e = 0
    for element in L:
        if element in garbage: continue
        count = 0
        t = 0
        if not e%100:
            print(f"Element {e+1}/{len(L)}")
        e +=1
        for transaction in T:   
            if False:
                print(f"Transaction {t+1}/{len(T)}")        
            c = int( set(element).issubset(transaction) )
            #print(f"Count of {element} in {transaction} = {c}")
            count += c
            t+= 1

        if count >= s:
            result.add(element)
        else: garbage.add(element)

    return result


def a_priori(T,s,k_max = 3):

    L1 = set()
    for transaction in T:
        for element in transaction:
            L1.add(element)

        
    print("Finished creating L1")
    

    L = [L1] + [None for _ in range(k_max)]
    C = [None for _ in range(k_max+1)]

    k = 1
    while len(L[k-1]) > 1:
        #print(f"Progress: {(k/set_size)*100:.3f} %")
        print(f"len = {len(L[k-1])}")
        #1. generate k-sized itemsets
        if k > 1:
            #print(f"k = {k}")
            C[k] = gen_Ck(L[1],k)
        else: C[k] = L1
        
        #print(f"C{k} = {C[k]}")
        print("Combinations done")

        # 2.Count the frequency of each itemset
        L[k] = count_item(T,C[k],s)

        #print(f"Count table: {count_table}")

        #L[k] = {element for element in count_table if count_table[element] >= s}

        #print(f"L{k} after = {L[k]}")
        k += 1
    
    return L[1:]


def gen_Ck(L1,k):

    return set(it.combinations(L1,k))

        

if __name__ == '__main__':

    transactions_file = "T10I4D100K.dat"

    T_db = []
    unique_elements = set()

    i = 0

    with open(transactions_file,"r") as T_file:

        T_file = T_file.readlines()

    
        for line in T_file:
            int_set = tuple(x for x in line.rstrip().split(" "))
            unique_elements.add(int_set)
            T_db.append(int_set)
            i+=1
    
    T = [("A", "D", "C"), ("B","C","E"),("A","B","C","E"),("B","E")]


    print("Finished reading file")

    pruned = a_priori(T_db,s=500)
    print(pruned)

#%%