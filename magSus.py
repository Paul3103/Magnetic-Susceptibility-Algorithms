from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time
import h5py

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from numpy.linalg import norm
import random

from jax import numpy as jnp
from jax import scipy as jscipy
from jax import grad, jacfwd, jit, vmap
from jax.lax import stop_gradient
from jax.config import config
from pylanczos import PyLanczos
import crystal as cry

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

    def __init__(self,fileName,temp):
        #Load values from speci
        global kB
        global uB
        global ge
        kB = 1.380649e-23  # Boltzmann constant in J/K
        uB = 9.2740100783e-24  # Bohr magneton in J/T
        ge = 2.00231930436  # electron g factor
        npValues = self.loadHDF5(fileName)
        self.setAngm(npValues[0])
        self.setHam(npValues[1])
        self.setSpin(npValues[2])
        self.setTemp(temp) #Temperature will need to be changed to a list of temperatures but for now I will only fopcs on temp =1.1 for debugging
        self.setEig(16)
        #self.calcZeeman(npValues)


    def getAngm(self):
        return self._angm
    
    def getHam(self):
        return self._H

        
    def getSpin(self):
        return self._spin
    
    def getEig(self):
        return self._eig
    def getTemp(self):
        return self._temp
    def getZeeman(self):
        return self._zeeman
    def setEig(self,newEig):
        self._eig = newEig

    def setAngm(self,newAngm):
        self._angm = newAngm

    def setHam(self,newHam):
        self._H = newHam

    def setSpin(self,newSpin):
        self._spin = newSpin
    def setTemp(self,newTemp):
        self._temp = newTemp
    def calcZeeman(self,spin, angm, field):
        '''Method for finding the zeeman hamiltonian <- need to do for the proper eigenvalues/vectors
        Inputs:
        Output: np.array <- The zeeman hamiltonain'''
        
        u = np.dot(uB,angm) + np.dot(ge,spin)
        Hzee = np.dot(u,field)
        self._zeeman = Hzee
        return Hzee

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
        Only worked for ops.hdf5, not 3gbOps.hdf5
        Method 5
        '''
        #print("davidson begins")
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
                #print("Converged in", iteration, "iterations")
            storedGuesses = new_guesses
        #sortedEigs = np.sort(eigenvalues)
        finish_david = time.time()
        #print("davidson = ", eigenvalues[:desiredEigs], ";", finish_david - start_david, "seconds")
        return np.sort(eigenvalues[:desiredEigs]), eigenvectors[:, :desiredEigs]

    def lanczos(self,matrix):
        '''
        Method to calculate Lanczos diagonalisation
        Method 6
        Returns eigenvalues and eigenvectors
        '''
        start_lanc = time.time() # Start timer
        #matrix = self.getHam()
        engine = PyLanczos(matrix, True, self.getEig())  # Find maximum eigenpairs
        eigenvalues, eigenvectors = engine.run()
        finish_lanc = time.time() # End timer
        #print("lanczos = ", eigenvalues[:self.getEig()], ";", finish_lanc - start_lanc, "seconds")
        map(np.real(eigenvalues))
        return eigenvalues[:self.getEig()], eigenvectors[:, :self.getEig()]
    
    def testEign(self,ham):
        '''
        Method is the numpy way of calculating eigenvalues/vectors
        also utilised the time.time() function to determine time taken for calculation

        returns E <- set of eigenvalues
        '''
        #print("numpy diagonalization begins")
        start_numpy = time.time()

        E,Vec = np.linalg.eigh(ham)
        E = np.sort(E)

        end_numpy = time.time()

        # End of Numpy diagonalization. Print results.

        #print("numpy = ", E[:self._eig],";",end_numpy - start_numpy, "seconds") 
        return E,Vec
    
    def loadHDF5(self,filename):
        '''
        method takes in a hdf5 file and reads it into the program

        Returns 2D array containing each array associated with a key
        '''

        with h5py.File(filename, "r") as selectedFile:
            # Print all aka keys 

            #print("Keys: %s" % f.keys())
            a_group_key = list(selectedFile.keys())[0]
            #print(selectedFile.keys())
            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            data = list(selectedFile[a_group_key])
            #print(a_group_key)

            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            data = list(selectedFile[a_group_key])
            # preferred methods to get dataset values:
            ds_obj = selectedFile[a_group_key]      # returns as a h5py dataset object
            ds_arr = selectedFile[a_group_key][()]  # returns as a numpy array

    
        return ds_arr

    def deriveH(self):
        #u . B <- (1 0 0) B0 <- B is just some vector, magnetism * B0
        # ^-- u = uB. angm + ge . spin (spin is a 3xNxN matrix)
        #dH/db is dHzeeman/ dBalpha because input hamiltonian is a constant == d/dBalpha [uxBx +uyBy + uzBz] == ualpha
        #Return [uxBx,uyBy,uzBz]
        u = np.dot(uB,self.getAngm()) + np.dot(ge,self.getSpin()) # doing u.B is unnecessary as the partial derivate will remove the Bx/By/Bz
        #print(u)
        ux = u[0]
        uy = u[1]
        uz = u[2]
        return [ux,uy,uz]


            
    def hellmanFeynman(self,dHdB,eigenvectors):
        first = np.dot(eigenvectors,dHdB)
        second = np.dot(first.T.conj(),eigenvectors)
        return second

    def calcMagSus(self, B, calc_eigs = 'lanczos'):
        '''
        Method to calculate magnetic susceptibility
        Inputs: B <- a float (magnetic field)
        Outputs: magSus <- magnetic susceptibility
        '''
        #Find Zeeman Hamiltonian
        #zeeman = self.calcZeeman(self.getSpin(),self.getAngm(),B)
        zeeman = self.zeeman_hamiltonian(self.getSpin(),self.getAngm(),B)
        #print("ZEEMAN FOR magsus")
        #print(zeeman)
        H = self.getHam() + zeeman
        fullham = self.getHam() + H
        #Calculate 
        if calc_eigs == 'lanczos':
            eigs = self.lanczos(fullham)
        elif calc_eigs == 'davidson':
            eigs = self.davidson(fullham)
        elif calc_eigs == 'numpy':
            eigs = self.testEign(fullham)
        else:
            print("Invalid")
            return 
        eigV = eigs[1]
        #print(eigs[0])
        magSus = []
        dim = self.getHam().shape[0]
        dHdB = self.deriveH()
        T = self.getTemp()

        for alpha in range(len(dHdB)): # Loop over x, y, z components
            dEdB = self.hellmanFeynman(dHdB[alpha], eigV)
            
            sum_dEdB_Z = 0
            sum_Z = 0

            for i in range(dim):
                exp_term = np.exp(self.getHam()[i] / (kB * T))
                sum_dEdB_Z += -(dEdB[i]) * exp_term
                sum_Z += exp_term

            magSus_component = sum_dEdB_Z / (B* sum_Z * uB)
            scalarV = np.sum(np.abs(magSus_component))
            #magSus.append(magSus_component)
            magSus.append(scalarV)

        return magSus


    def calculate_magnetism(self):
        '''
        Method to help understand how to calculate susceptibility
                     dim 
            1        ____
          ___   *   \      dEi/dBalpha * e^(-Ei/kBT)
          Z*uB      /
                   /___
                    i = 1
          
        '''

    def magmom(self,spin, angm):
        muB = 0.5  # atomic units
        g_e = 2.002319
        return muB * (angm + g_e * spin)

    def zeeman_hamiltonian(self,spin, angm, field):
        """Compute Zeeman Hamiltonian in atomic units.

        Parameters
        ----------
        spin : np.array
            Spin operator in the SO basis.
        angm : np.array
            Orbital angular momentum operator in the SO basis.
        field : np.array
            Magnetic field in mT.

        Returns
        -------
        np.array
            Zeeman Hamiltonian matrix.
        """

        au2mT = 2.35051756758e5 * 1e3  # mTesla / au

        # calculate zeeman operator and convert field in mT to T
        return (jnp.array(field) / au2mT)* self.magmom(spin, angm)



fileName = "ops.hdf5"
temperatures1 = [1.1]

field = 0.8


mag = magSusCalculator(fileName,1.1)

#print(mag.testEign())
#print(mag.lanczos()[0])
#print(mag.davidson()[0])

angmomSus = cry.MagneticSusceptibilityFromFile(fileName,temperatures=temperatures1,field=0.8 , differential = True)
answer = mag.calcMagSus(field,calc_eigs='numpy')  
#answer = mag.calcMagSus(field,calc_eigs='lanczos')  

#print(answer)
print(angmomSus.evaluate())
#print(np.sum(np.abs(answer)))
#print(angmomSus.evaluate())
'''
mat1 = [-2.95397665e-03, -2.95397664e-03, -9.92770333e-04, -9.92770226e-04
  ,4.95265690e-05,  4.95273956e-05,  5.15419136e-04,  5.15424715e-04
  ,6.38918084e-04, 6.38925510e-04 , 7.67017994e-04 , 7.67018442e-04
  ,9.32656389e-04,  9.32657495e-04,  1.04319908e-03,  1.04320303e-03]
mat2 = [-2.95397665e-03, -2.95397664e-03, -9.92770333e-04, -9.92770226e-04
  ,4.95265690e-05,  4.95273956e-05,  5.15419136e-04,  5.15424715e-04
  ,6.38918084e-04,  6.38925510e-04,  7.67017994e-04,  7.67018442e-04
  ,9.32656389e-04 , 9.32657495e-04 , 1.04319908e-03  ,1.04320303e-03]
for i in range(0,len(mat1)):
    print(mat1[i]==mat2[i])
'''