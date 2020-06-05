'''
SVD OF GIVEN MATRICES
===========================================================
Author :  Rounak CHatterjee
Date : 05/06/2020
===========================================================
To Compute the singular value decomposition of a matrix

we need to find two matrices U and V sucht that

S = U^T.A.V, where S is the Singular decomposed matrix. These Matrices
can ne fount by doing an Eigen analysis of A.A^T and A^T.A for which
 U and V are the similarity transformers. Lickily for us numpy has 
 a module named numpy.linalg.svd that cn irectly compute the svd of a Matrix.
 We will us that in this code. 

 The SVD package computes U,V and S and outpus The matrix U followed by diagonal elements of S and then by V.
 We will us these facts.
 '''

import numpy as np
import numpy.linalg as l
# we initialize both the arrays
A_1 = np.array([[2,1],[1,0],[0,1]],dtype = np.float64)
A_2 = np.array([[1,1,0],[1,0,1],[0,1,1]],dtype = np.float64)

#Doing SVD 

S_1 = l.svd(A_1)
S_2 = l.svd(A_2)

#Printing
print("First Matrix : \n",A_1,"\nThe Singular Values are : \n",S_1[1])
print("Second Matrix : \n",A_2,"\nThe Singular Values are : \n",S_2[1])
