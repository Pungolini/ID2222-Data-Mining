# %%
import numpy as np
import random
import time
import itertools as it

class Shingling:

	def __init__(self):
		pass

	def k_shingles(self, docs : list, k=1) -> list:
		"""
		Implements k-shingles algorithm on documents passed by argument.
		docs: Documents, a list of lists of strings. A document is a iterable (a)
		Returns: a set of k-sized tokens. Tokens in this function are a sequence of characters instead of words
		"""
		shingle_set = []
		for doc in docs:
			doc_shingle = [doc[i:i + k] for i in range(len(doc) - k)]
			shingle_set.append(dict.fromkeys(doc_shingle).keys())  # Trick to emulate an ordered set
		return shingle_set

	def hash_shingle(self, shingles):

		return dict.fromkeys(hash(token) for token in shingles).keys()

	def create_documents(self, text_list, size):

		# create the set of character sequence of length k=size
		list_of_shingles = self.k_shingles(text_list, k=size)

		# Compute the hash of each sequence of character
		documents = [ self.hash_shingle(char_set) for char_set in list_of_shingles]#ATT: dont forget to put back self.hash_shingle(char_set)
		return documents

class CompareSets:

	def __init__(self, set_A:set, set_B:set) -> None:
		self.set_A = set_A
		self.set_B = set_B

	def jaccard_similarity(self) -> float:
		"""
		Computes the Jaccard Similarity of A and B
		A,B : sets
		returns: int, the Jaccard Similarity
		jaccard_sim = overlapping/total_items = [ set(A)&set(B)] / len(set(A)+set(B)) = TP/(TP+FP+FN)

		"""
		A = self.set_A
		B = self.set_B
		if not isinstance(A,set): A = set(A)
		if not isinstance(B,set): B = set(B)


		overlapping = A & B
		total = A | B
		assert total
		return round(len(overlapping) / len(total),4)

	def jaccard_distance(self) -> float:
		return round(1 - self.jaccard_similarity(),4)

class CompareSignatures:

	def __init__(self,sig_A: list,sig_B: list) -> None:
		self.sig_A = np.array(sig_A)
		self.sig_B = np.array(sig_B)

		#print(sig_A)
		print("--------------------------------------------")
		#print(sig_B)

		print(f"Signature similarities: {self.compare_sigs()}")

	def compare_sigs(self) -> float:

		sim = 0
		assert len(self.sig_A) == len(self.sig_B)

		for i in range(len(self.sig_A)):
			for j in range(len(self.sig_B)):

				sim += self.sig_A[i] == self.sig_B[j]
		
		print(sim,sim/len(self.sig_A))

		assert len(self.sig_A) == len(self.sig_B)

		return sim/len(self.sig_A)


class MinHash:

	def __init__(self, *documents: list, k: int) -> None:
		self.docs = [doc for doc in documents]
		self.k = k
		self.table= []
		self.c = 4294967311
		random.seed(time.time())

	def min_hash(self,n=100):


		# Generate n paires of unique random (a,b) coefficients
		hashes = [(random.randint(0,self.c-1),random.randint(0,self.c -1)) for _ in range(n)]

		# Get list of individual shingles
		all_shingles, list_of_docs = self.get_universal_shingles()

		bool_table = np.array(self.build_bool_table(all_shingles,list_of_docs)).T

		sig_matrix = []
		x = []
		i = -1
		for shingle_set in list_of_docs:
			i += 1
			signature_ = self.get_sig(hashes,shingle_set,n)
			signature = self.calc_signature_vector(list(shingle_set),hashes,bool_table[i],n)
			sig_matrix.append(signature)
			x.append(signature_)

			#print(signature)

		
		return sig_matrix,x


	def calc_signature_vector(self,shingle_set,hashes,bool_vect,n=10):

		# Initialize signature vector's elements as infinity
		signature = [inf for _ in range(n)]

		for i in range(len(shingle_set)):
			shingle_in_doc = bool_vect[i] == 1
			if shingle_in_doc:
				for j in range(n):
					a,b = hashes[j]
					hash_val = (a*shingle_set[i] + b)%self.c
					if signature[j] > hash_val:
						signature[j] = hash_val
		return signature

	
	def get_sig(self,hashes,shingles,n=100):

		sig = [inf for _ in range(n)]

		for i in range(n):

			for shingle in shingles:
				a,b = hashes[i]
				h = (a*shingle + b)%self.c

				if h < sig[i]:
					sig[i] = h
		return sig



	def build_bool_table(self,all_shingles,list_of_docs) -> list:

		"""
		no_duplicates: list of unique shingles from all documents -> set of strings

		list_of_docs: list of all documents, each document represented by the 
					  hashes of their k-strings
		
		returns: a boolean table where the rows represents each unique shingle
		and the columns are the documents. table[i][j] = 1 if shingle_1 is in document j
		"""

		table = []
		for shingle in all_shingles:
			row = [int(shingle in doc) for doc in list_of_docs]
			table.append( row )
			
		self.table = table
		
		return table


	def hash_function(self,x: int) -> int: # row is a row
		a = random.randint(0,self.c -1)
		b = random.randint(0,self.c -1)
	
		return (a*x+b)%self.c


	def similarity(self, table) -> int:

		# Type A rows, ie, rows where all elements are "1"s
		inter = len(np.where(sum(table.T) == 2)[0])

		# Type B and C rows, ie, rows where at least a "1" is present
		union = len(np.where(sum(table.T) >= 1)[0])

		assert union > 0
		return round(inter/union,2)

	def get_universal_shingles(self) -> tuple:
		"""
		Method that returns the set of unique hashed k-shingles for all documents in
		self.docs and the list of documents (list of hashed k-shingles)
		"""

		# For each document, hash its shingles and return  the list of docs
		list_of_docs = Shingling().create_documents(self.docs, size=self.k)

		# Remove duplicates among all shingles from all documents
		no_duplicates = []
		for shingle_set in list_of_docs:
			for char_seq in shingle_set:
				no_duplicates.append(char_seq)

		no_duplicates = list(dict.fromkeys(no_duplicates))

		#no_dupli = list(list(k) for k,_ in it.groupby(list_of_docs))

		return no_duplicates,list_of_docs

		#print('No duplicates', no_duplicates)

	# print(list_of_shingles)

	# documents = S.create_documents(self.documents,size=self.k)


if __name__ == '__main__':
	from pprint import pprint
	from math import inf
	C1 = [0, 1, 1, 0, 1, 0]
	C2 = [1, 0, 1, 0, 1, 1]
	D1 = "I am Sam"
	D2 = "Sam I am"
	D3 = "I do not like green eggs and ham"
	D4 = "I do not like them Sam I am"
	D = "abcab"
	Docs = (D1, D2, D3)


	#################################################################


	P = 10 # Number of permutations

	data1 = ''.join(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
			'estimating', 'the', 'similarity', 'between', 'datasets'])
	data2 = ''.join(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
			'estimating', 'the', 'similarity', 'between', 'documents'])

		
	M = MinHash(data1,data2, k=3)

	sig,x = M.min_hash()

	CompareSignatures(*sig)
	CompareSignatures(*x)
	print("*====================================")



#%%
# print(table,M.intersection(table),M.union(table),M.similarity(table))

# Shingling and CompareSets tests
if True:
	S = Shingling()

	D1 = "I am Sam"
	D2 = "Sam I am"
	D3 = "I do not like green eggs and ham"
	D4 = "I do not like them Sam I am"
	D = "abcab"

	DD1 = "editorial"
	DD2 = "factorial"

	M = MinHash(D4,D3, k=3)

	sig,x = M.min_hash()

	CompareSignatures(*sig)
	CompareSignatures(*x)

# doc1 = S.k_shingles(D1,D2,D3,k=2)

# print(S.create_documents(D1,D2,D3,size=2))
# doc2 = S.k_shingles_faster(DD2,k=9)
# print(doc1)
# print(CompareSets(doc1,doc2).jaccard_similarity())

# print(f"k = {2}  s = {s}    hash = {S.hash_shingle(s)}")

# %%