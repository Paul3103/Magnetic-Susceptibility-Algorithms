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
        self.setEig(16)


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

    def davidson(self, tol=1e-8, max_iterations=100):
        '''
        Method not just borrowed from online
        Returns eigenvalues and eigenvectors
        Method 5
        '''
        start_david = time.time()

        useHam = self.getHam()
        desiredEigs = self.getEig()
        hDim = useHam.shape[0]  # 16x16, get first dimension
        unitVectors = np.eye(hDim, desiredEigs)  # Set of unit vectors
        storedGuesses = np.zeros((hDim, desiredEigs))  # Empty array which will hold guesses
        identifyHam = np.eye(hDim)  # Identity matrix for the Hamiltonian

        initial = True
        accurate = False
        iteration = 0

        while not accurate and iteration < max_iterations:
            iteration += 1
            if initial:
                initial = False
                for j in range(0, desiredEigs):
                    storedGuesses[:, j] = unitVectors[:, j] / np.linalg.norm(unitVectors[:, j])
            else:
                for j in range(0, desiredEigs):
                    matVecResult = np.dot(useHam, storedGuesses[:, j])
                    storedGuesses[:, j] = matVecResult / np.linalg.norm(matVecResult)
                    
            # Form the subspace matrix
            subspace_matrix = np.dot(storedGuesses.T, np.dot(useHam, storedGuesses))
            eigenvalues, eigenvectors = np.linalg.eigh(subspace_matrix)
            
            # Form improved guesses
            new_guesses = np.dot(storedGuesses, eigenvectors)
            
            # Check convergence
            conv_check = np.max(np.abs(new_guesses - storedGuesses))
            if conv_check < tol:
                accurate = True
            storedGuesses = new_guesses
        #sortedEigs = np.sort(eigenvalues)
        finish_david = time.time()
        print("davidson = ", eigenvalues[:desiredEigs], ";", finish_david - start_david, "seconds")
        return eigenvalues[:desiredEigs], eigenvectors[:, :desiredEigs]

    def lanczos(self, tol = 1e-8, max_iterations = 100):
        '''
        Method to calculate Lanczos diagonalisation
        Method 6
        Returns eigenvalues and eigenvectors
        '''
        start_lanc = time.time() # Start timer
        #TODO: Implement Lanczos diagonalisation
        pass

        finish_lanc = time.time() # End timer
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

        print("numpy = ", E[:self._eig],";",end_numpy - start_numpy, "seconds") 
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

mag.davidson()[0]
mag.testEign()[:mag.getEig()]
#print(mag.jaxianApproach()[:4])
#print(mag.lanczos(mag.getHam,4))

