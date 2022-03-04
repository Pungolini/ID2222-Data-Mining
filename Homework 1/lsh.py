import itertools as it
class LSH:

    def lsh_pairs(self,n_bands,n_rows,sig_mat):
        n = n_bands*n_rows
        buckets = dict([])
        lsh_pairs_set = {}

        for i in range(n_bands):

            s = i*n_rows
            e = s + n_rows
            single_band = sig_mat[s:e]

            k = 0
            for col in single_band.T:
                key = hash(tuple(col))
                buckets[key].append(k)
                k += 1
            
            for docs in buckets.values():
                pairs = it.combinations(docs,2)

                for pair in pairs:
                    lsh_pairs_set.add(pair)
                
            buckets = {}
        
        return lsh_pairs_set

    
    def compare_docs(self,sig_mat,threshold,):

        # find the candidate pairs by applying LSH
        lsh_pairs_set = self.lsh_pairs(sig_mat)
        near_duplicates = set()

        for a,b in lsh_pairs_set:

            similarity = CompareSignatures(a,b).compare_sigs()
            if similarity > threshold:
                near_duplicates.add((a,b))

        return near_duplicates
    