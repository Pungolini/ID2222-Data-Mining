# %%
import argparse
import numpy as np
import random
import time
import collections
from itertools import combinations



def Shingling(doc: str, k=4) -> list:
	"""
	Implements k-shingles algorithm on documents passed by argument.
	doc: Document, represented as a big string.
	Returns: a set of k-sized tokens. Tokens in this function are a sequence of characters instead of words
	"""
	list_of_shingles = dict.fromkeys(doc[i:i + k] for i in range(len(doc) - k + 1)).keys()

	return dict.fromkeys(hash(token) for token in list_of_shingles).keys()


class CompareSets:

	def __init__(self, set_A: set, set_B: set) -> None:
		self.set_A = set_A
		self.set_B = set_B
		print(f"The Jaccard similarity is {self.jaccard_similarity()}")
		print(f"The Jaccard distance is {self.jaccard_distance()}")

	def jaccard_similarity(self) -> float:
		"""
		Computes the Jaccard Similarity of A and B
		A,B : sets
		returns: int, the Jaccard Similarity
		jaccard_sim = overlapping/total_items = [ set(A)&set(B)] / len(set(A)+set(B)) = TP/(TP+FP+FN)

		"""
		A = self.set_A
		B = self.set_B
		if not isinstance(A, set): A = set(A)
		if not isinstance(B, set): B = set(B)

		overlapping = A & B
		total = A | B
		assert total
		return round(len(overlapping) / len(total), 4)

	def jaccard_distance(self) -> float:
		return round(1 - self.jaccard_similarity(), 4)


class CompareSignatures:

	def __init__(self, sig_A: list, sig_B: list) -> None:
		self.sig_A = np.array(sig_A)
		self.sig_B = np.array(sig_B)
		sim = self.compare_sigs()

		print(f"Signature similarities: {sim:.4f}")

		if sim > 0.7:
			print("The documents are similar.")
		else:
			print("The documents are not similar.")

	def compare_sigs(self) -> float:

		sim = 0
		assert len(self.sig_A) > 0
		assert len(self.sig_A) == len(self.sig_B)

		for i in range(len(self.sig_A)):
			for j in range(len(self.sig_B)):
				sim += self.sig_A[i] == self.sig_B[j]

		return round(sim / len(self.sig_A), 4)


class MinHash:

	def __init__(self, *documents: list, k: int) -> None:
		self.docs = [doc for doc in documents]
		self.k = k
		self.table = []
		self.c = 4294967311
		random.seed(time.time())

	def create_sig_matrix(self, n=100):

		# Generate n paires of unique random (a,b) coefficients
		hashes = [(random.randint(0, self.c - 1), random.randint(0, self.c - 1)) for _ in range(n)]

		# Get list of individual shingles
		all_shingles, list_of_docs = self.get_universal_shingles()

		bool_table = np.array(self.build_bool_table(all_shingles, list_of_docs)).T

		sig_matrix = []
		for c in range(len(list_of_docs)):
			signature = self.calc_signature_vector(c, list_of_docs, hashes, bool_table, n)
			sig_matrix.append(signature)

		return sig_matrix

	def calc_signature_vector(self, column, list_of_docs, hashes, bool_table, n):

		bool_vect = bool_table[column]
		shingle_set = list(list_of_docs[column])

		# Initialize signature vector's elements as infinity
		signature = [float('inf') for _ in range(n)]

		for i in range(len(shingle_set)):
			shingle_in_doc = bool_vect[i] == 1
			if shingle_in_doc:
				for j in range(n):
					a, b = hashes[j]
					hash_val = (a * shingle_set[i] + b) % self.c
					signature[j] = min(hash_val, signature[j])
		return signature

	def build_bool_table(self, all_shingles, list_of_docs) -> list:

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
			table.append(row)

		self.table = table

		return table

	def similarity(self, table) -> int:

		# Type A rows, ie, rows where all elements are "1"s
		inter = len(np.where(sum(table.T) == 2)[0])

		# Type B and C rows, ie, rows where at least a "1" is present
		union = len(np.where(sum(table.T) >= 1)[0])

		assert union > 0
		return round(inter / union, 2)

	def get_universal_shingles(self) -> tuple:
		"""
		Method that returns the set of unique hashed k-shingles for all documents in
		self.docs and the list of documents (list of hashed k-shingles)
		"""

		# For each document, hash its shingles and return  the list of docs
		list_of_docs = [Shingling(doc, k=self.k) for doc in self.docs]

		# Remove duplicates among all shingles from all documents
		no_duplicates = []
		for shingle_set in list_of_docs:
			for char_seq in shingle_set:
				no_duplicates.append(char_seq)

		no_duplicates = list(dict.fromkeys(no_duplicates))
		# no_dupli = list(list(k) for k,_ in it.groupby(list_of_docs))
		return no_duplicates, list_of_docs


class LSH:

	def __init__(self,n_bands=20,n_rows=5):
		self.n_bands = n_bands
		self.n_rows = n_rows

	def lsh_pairs(self,sig_mat):
		n_rows = self.n_rows
		n_bands = self.n_bands
		n = n_bands*n_rows

		buckets = collections.defaultdict(list)
		lsh_pairs_set = {}

		for i in range(n_bands):

			s = i*n_rows
			e = s + n_rows
			single_band = np.array(sig_mat[s:e])

			k = 0
			for col in single_band.T:
				key = hash(tuple(col))
				buckets[key].append(k)
				k += 1
			
			for docs in buckets.values():
				pairs = combinations(docs,2)

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

if __name__ == '__main__':

	#################################################################

	parser = argparse.ArgumentParser(description='Argument Parser. Please choose text 1 and 2, the'
												 'shingle length and the number of permutations')

	parser.add_argument('--text1', default='dogs_wiki.txt', choices=['lorem_ipsum1.txt','lorem_ipsum2.txt','lorem_ipsum3.txt','dogs_wiki.txt', 'cats_wiki.txt', 'lorem_1.rtf',
																	 'lorem_2.rtf', 'church_1.txt', 'church_2.txt'])
	parser.add_argument('--text2', default='cats_wiki.txt', choices=['lorem_ipsum1.txt','lorem_ipsum2.txt','lorem_ipsum3.txt','dogs_wiki.txt', 'cats_wiki.txt', 'lorem_1.rtf',
																	 'lorem_2.rtf', 'church_1.txt', 'church_2.txt'])
	parser.add_argument('--shingle_len', default=5, type=int)
	parser.add_argument('--permutations', default=100, type=int)

	args = parser.parse_args()

	with open('Texts/' + args.text1, 'r') as file:
		data1 = file.read()

	with open('Texts/' + args.text2, 'r') as file:
		data2 = file.read()

	data1 = data1.replace('\n', '').replace('\r', '')
	data1 = data1.replace(' ', '')
	data2 = data2.replace('\n', '').replace('\r', '')
	data2 = data2.replace(' ', '')

	k = args.shingle_len  # shingle length
	P = args.permutations  # Number of permutations

	s1 = Shingling(data1, k)
	s2 = Shingling(data2, k)
	CompareSets(s1, s2)

	M = MinHash(data1, data2, k=k)

	sig = M.create_sig_matrix()

	CompareSignatures(sig_A=sig[0], sig_B=sig[1])

	similar_docs = LSH().compare_docs(sig,0.2)

