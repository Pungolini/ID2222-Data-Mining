#!/usr/bin/env python
# coding: utf-8

# In[833]:


import numpy as np
import networkx as nx
from numpy.linalg import norm ,inv,eigh
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


# In[834]:
def read_points(filename):

	points = []

	with open("data/" + filename, 'r') as dataset:

		for line in dataset:
			point = [int(x) for x in line.split(',')[:2]]

			points.append(point)
	return np.array(points)

# In[835]:

def build_A(points,unweighted = True,sigma_sq=100):

	n = len(points)
	A = np.zeros((n,n))

	weight = 1
	for i in range(n):
		si = points[i]
		for j in range(n):

			if i <= j: continue # Matrix is symmetric

			sj = points[j]

			if not unweighted:
				dist = norm(si-sj)**2
				weight = np.exp(-dist/(2*(sigma_sq)))

			A[i][j] = A[j][i] = weight
	return A


def build_L(A,D):

	sqrt_D = inv(sqrtm(D))

	return sqrt_D.dot(A).dot(sqrt_D)


# In[840]:


# In[844]:

def build_D(A):

	n = len(A)

	D = [np.sum(A[i]) for i in range(n)]

	return np.diag(D)



if __name__ == '__main__':

	import argparse
	import time

	start = time.time()

	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Argument Parser. Please choose the graph dataset and the number of partitions, K')

	parser.add_argument('--dataset', default="example1.dat",choices=["example1.dat", "example2.dat"])
	parser.add_argument('--partitions', default=4, type=int)

	args = parser.parse_args()

	dataset_file = args.dataset
	K = args.partitions

	# Read points from the dataset file

	# Read edges from file
	edges = read_points(dataset_file)

	# Create Graph Object
	G = nx.Graph() 
	G.add_edges_from(edges) # Add edges from file content

	print("1. Finished building the graph\n")

	# Compute matrices A,D and L
	A = nx.adjacency_matrix(G).toarray() 	# Get Adjacency matrix in dense format
	D = build_D(A)
	L = build_L(A,D)

	print(f"2. Finished building A,D,L\n")


	# Get the eigenvalues and vectors sorted (it's sorted in descending order)
	eigvals,eigvects = eigh(L)

	print(f"3. Finished computing eigenvalues\n")


	# Stack the K largest eigenvectors into matrix X
	X = eigvects[:,-K:]

	# Y is the row-normalized version of X
	Y = X/norm(X, axis = 0, keepdims = True)

	#print(np.sum( Y**2, axis=0 )) # Uncomment to check that the normalization is correct

	print(f"4. Finished computing X and Y\n")


	# Run KMeans algorithm
	labels = KMeans(n_clusters= K).fit_predict(Y)

	print(f"5. Finished clustering\n")
	for clust in np.unique(labels):
		print(f"Cluster {clust} has  {len(np.where(labels == clust)[0])} nodes")
	print()


	end = time.time()

	print(f"Execution time: {end - start:.2f} seconds\n")
	############ PLOTS ############################

	# Draw colored clusters
	nx.draw(G, node_size=30,node_color = labels)

	plt.savefig(f"results/{dataset_file[:-4]}_clusters")
	num_nodes = X.shape[0]


	# Eigenvector associated to the 2nd largest eigenvalue
	v2 = eigvects[num_nodes - 2]
	fiedler_vect = sorted(v2)

	# Plot Fiedler vector and eigenvalues
	fig, axs = plt.subplots(ncols = 2,sharex = True,sharey = False)
	axs[0].plot(sorted(eigvals))
	axs[0].set_title('Eigenvalues of Laplacian $L_{Graph}$')
	axs[1].plot(fiedler_vect)
	axs[1].set_title('Fiedler vector ')
	axs[0].grid()
	axs[1].grid()
	plt.savefig(f"results/{dataset_file[:-4]}_spectrum.png")

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()

	plt.show()

	# Plot the Sparsity of A
	plt.title("Sparsity of Matrix A")

	plt.spy(A)
	plt.savefig(f"results/{dataset_file[:-4]}_sparsity.png")
	plt.show()
