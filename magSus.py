from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time
import h5py

from jax import numpy as jnp
from jax import scipy as jscipy
from jax import grad, jacfwd, jit, vmap
from jax.lax import stop_gradient
from jax.config import config


class magSusCalculator:
    '''
     magSusCalculator is the class which will have methods pertaining to each method described below:
    1) Jakob's original JAX/python implementation that vectorises over temperatures - "fast" but memory hog.
    2) Jakob's revised JAX/python implementation that splits temperatures into a loop - slower, but much smaller memory footprint
    3) Use JAX in python to calculate eigenvalues and derivatives "naively", then do a simple loop over T
    4) Get eigenvalues and then use Hellman-Feynman theorem to get eigenvectors, then do a simple loop over T - NFC will implement this in Fortran for comparison
    5) Implementation of 4 using Davidson diagonalisation
    6) Implementation of 4 using Lancoz diagonalisation
    7) Reformulating the expression as a matrix exponential and trying various approximations.....
    '''

    def __init__(self,fileName):
        #Load values from speci
        npValues = self.loadHDF5(fileName)
        self.setAngm(npValues[0])
        self.setHam(npValues[1])
        self.setSpin(npValues[2])
        self.setEig(4)


    def getAngm(self):
        return self._angm
    
    def getHam(self):
        return self._H
    def getSpin(self):
        return self._spin
    
    def getEig(self):
        return self._eig

    def setEig(self,newEig):
        self._eig = newEig

    def setAngm(self,newAngm):
        self._angm = newAngm

    def setHam(self,newHam):
        self._H = newHam

    def setSpin(self,newSpin):
        self._spin = newSpin

    def jaxianApproach(self):
        '''
	Method 1: Jakobs Original implementation using JAX
        Method modified from angmom_suite/crystal.py source code
        Code removed will be reimplemented when required (Actually doing calcs with eigenvalues)
        '''
    
        eig, vec = jnp.linalg.eigh(self.getHam())
        labs = np.unique(np.around(eig, 8), return_inverse=True)[1]
        #print(labs)
        return eig



    def davidsonD(self):
        '''
        Method 6: Do method 3 using Davidson diagonalisation to find the eigenvalues
        
        Credit to James Going: https://joshuagoings.com/2013/08/23/davidsons-method/
        Block Davidson method for finding the first few
        lowest eigenvalues of a large, diagonally dominant, sparse Hermitian matrix (e.g. Hamiltonian)
        Currently not calculating accurate eigenvalues for sample 16x16 matrix
        '''
        
        useHam = self.getHam()  # This is the matrix to be calculated
        #useHam = np.real(useHam)

        n = useHam.shape[0]  # Dimension of matrix, assuming it's a square matrix
        tol = 1e-10  # Convergence tolerance
        k = 8  # Number of initial guess vectors
        eig = self.getEig()  # Number of eigenvalues to solve
        mmax = max(k * 20, n)  # Maximum number of iterations, use a reasonable value

        t = np.eye(n, k)  # Set of k unit vectors as guess
        V = np.zeros((n, n), dtype=complex)  # Array of zeros to hold guess vectors
        I = np.eye(n)  # Identity matrix same dimension as useHam

        # Begin block Davidson routine
        start_davidson = time.time()

        for m in range(k, mmax, k):
            if m <= k:
                for j in range(0, k):
                    V[:, j] = t[:, j] / np.linalg.norm(t[:, j])
                theta_old = 1
            else:
                theta_old = theta[:eig]

            V[:, :m], R = np.linalg.qr(V[:, :m])
            T = np.dot(V[:, :m].T, np.dot(useHam, V[:, :m]))  # Use the provided "useHam" matrix
            THETA, S = np.linalg.eig(T)
            idx = THETA.argsort()
            theta = THETA[idx]
            s = S[:, idx]
            for j in range(0, k):
                w = np.dot((useHam - theta[j] * I), np.dot(V[:, :m], s[:, j]))  # Use "useHam" matrix
                q = w / (theta[j] - useHam[j, j])  # Use "useHam" matrix
                V[:, (m + j)] = q
            norm = np.linalg.norm(theta[:eig] - theta_old)
            if norm < tol:
                break

        end_davidson = time.time()

        # End of block Davidson. Print results.
        E = theta[:eig]
        E = np.sort(E)
        print("davidson = ", theta[:eig], ";", end_davidson - start_davidson, "seconds")

      
        return E

   

# Example usage:
# Define your sparse symmetric matrix A and the number of eigenvalues k to find
# eigvals, eigvecs = lanczos(A, k)

    def testEign(self):
        '''
        Method is the numpy way of calculating eigenvalues/vectors
        also utilised the time.time() function to determine time taken for calculation

        returns E <- set of eigenvalues
        '''
        
        start_numpy = time.time()

        E,Vec = np.linalg.eig(self.getHam())
        E = np.sort(E)

        end_numpy = time.time()

        # End of Numpy diagonalization. Print results.

        #print("numpy = ", E[:self._eig],";",end_numpy - start_numpy, "seconds") 
        return E
    
    def loadHDF5(self,filename):
        '''
        method takes in a hdf5 file and reads it into the program

        Returns 2D array containing each array associated with a key
        '''

        with h5py.File(filename, "r") as selectedFile:
            # Print all aka keys 

            #print("Keys: %s" % f.keys())
            a_group_key = list(selectedFile.keys())[0]


            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            data = list(selectedFile[a_group_key])


            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            data = list(selectedFile[a_group_key])
            # preferred methods to get dataset values:
            ds_obj = selectedFile[a_group_key]      # returns as a h5py dataset object
            ds_arr = selectedFile[a_group_key][()]  # returns as a numpy array

    
        return ds_arr


    




mag = magSusCalculator("ops.hdf5")

mag.davidsonD()
#print(mag.davidsonD()[:4])
print(mag.testEign()[:4])
#print(mag.jaxianApproach()[:4])
#print(mag.lanczos(mag.getHam,4))

