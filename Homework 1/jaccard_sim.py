

def j_s(A,B):
    """
    Computes the Jaccard Similarity of A and B
    A,B : list/tuple
    returns: int, the Jaccard Similarity
    jaccard_sim = overlapping/total_items = [ set(A)&set(B)] / len(set(A)+set(B)) = TP/(TP+FP+FN)

    """
    A = set(A)
    B = set(B)
    overlapping = A&B
    total = A | B
    return len(overlapping)/len(total)