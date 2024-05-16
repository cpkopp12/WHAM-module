"""
author: Cameron Kopp

created: 5/15/2024
"""
#IMPORTS
import numpy as np
import scipy.integrate as integrate
from numpy import random as r
import math
import matplotlib.pyplot as plt
from matplotlib import colors

# =============================================================================
# WhamConvergence_1Ion class
# =============================================================================

class WhamConvergence_1Ion:
    """
    class contains all of the methods for solving the WHAM Equations,
    going to begin by using a BFGS convergence algoritm with an Armijo
    line search
    """
    def __init__(self, xmin, xmax, biasingStepNumber, sampleNumber, binSize, springConstant,
                 fname):
        """
        

        Parameters
        ----------
        xmin : float
            min value on range xmin-xmax
        xmax : float
            max value on ranage xmin-xmax.
        biasingStepNumber : Int
            determines the number of sets of samples taken from different 
            biasing centers, evenly spaced over xmin-xmax.
        binSize : float
            size of the bins which discriteze the x coord.
        springConstant : float
            determines the strength of the biasing potential.
        
        fname : string
            name of .txt file to read data from

        Returns
        -------
        going to create arrays for bin indices and coordinates, and well as
        biasing indicies and coordinates
        going to read in and instantiate histogram from the datafile
        
        also establishes the neccessary params which will not be altered
        throughout convergence, i.e. biasing potential, summations from 
        histogram
        

        """
        
        self.xmn = xmin
        self.xmx = xmax
        self.bStepNum = biasingStepNumber
        self.N = sampleNumber
        self.k = springConstant
        self.binSize = binSize
        self.fileName = fname
        
        
        #set up two arrays, bin coordinates (center) and bin indicies
        binNumber = (self.xmx-self.xmn)/self.binSize
        #array with bin indexes = binI
        self.binI = np.arange(0,binNumber, dtype=int)
        #array with bin x coordinate values = binX
        self.binX = np.array((self.binI * self.binSize) + (self.binSize/2),
                               dtype=float)
        
        #set up two arrays, coordinates of the center of biasing potential
        #for each step biasX, and the indicies for the biasing steps baisI
        stepSize = (self.xmx-self.xmn)/(self.bStepNum + 1)
        self.biasI = np.arange(0, self.bStepNum, dtype=int)
        self.biasX = np.array((self.biasI * stepSize) + stepSize,
                                dtype=float)
        
        #histogram array
        self.hist = np.zeros(np.size(self.binI))
        
        #need total number of samples per baising step
        self.N_i = int(math.floor(self.N/self.bStepNum))
        
        #going to call the baising potential matix c_i_l (dim[SxM])
        self.c_i_l = self.biasingPotentialMatrix()
        
        
        #read histogram data from file
        with open(self.fileName) as myFile:
            st= myFile.read();
            
        lst = st.split()
        for i in self.binI:
            self.hist[i] = float(lst[i])
            
            
    def printHistogramFromFile(self):
        # make data
        x = self.binI
        y = self.hist
         
        # plot
        fig, ax = plt.subplots()
         
        ax.plot(x, y, linewidth=2.0)
         
        plt.show()
        return
    
    def biasingPotentialMatrix(self):
        """
        calculate the biasing potential matrix, dim(size(biasI),size(binI)),
        the element i,l represents the value of the biasing potential from
        simulation step i at the x value in the center of bin l

        Returns
        -------
        biasing potential matrix, dim(size(biasI),size(binI)), dtype=float

        """
        
        c_i_l = np.zeros((np.size(self.biasI),np.size(self.binI)),
                                       dtype = float)
        
        #loop over bias index, get x coord for each
        for i in self.biasI:
            iIn = int(i)
            biasx = self.biasX[iIn]
            #loop over bin index, get x coord for each
            for l in self.binI:
                lIn = int(l)
                binx = self.binX[lIn]
                #calc elment of bias matrix
                c_i_l[iIn, lIn]  = np.exp((-1*(self.k/2))*((binx- biasx)**2))
                 
                
        return c_i_l
    
    def initialEstimates(self):
            """
            initial estimations for rho, normalization constant f,
            and the change of variable g = log(1/f),
            going to use the most simple approximation possible,
            each vector will have identical elemnts independent of
            histogram data

            Returns
            -------
            (p0, f0, g0)

            """
            p0 = np.zeros((np.size(self.binI)),dtype=float)
            f0 = np.zeros((np.size(self.biasI)),dtype=float)
            g0 = np.zeros((np.size(self.biasI)),dtype=float)
            
            #equal elements N/bin number for p0
            binN = (self.xmx-self.xmn)/self.binSize
            p0el = self.N/binN
            p0 = p0+p0el
            
            #f0 will be the sum of 1/(p0(element)) over the number
            #of bins in each biasing step size
            stepSize = (self.xmx-self.xmn)/(self.bStepNum + 1)
            binsPerStep = stepSize/self.binSize
            psum = p0el*binsPerStep
            f0 = f0 + 1/psum
            g0 = g0 + np.log(f0)
            
            return (p0, f0, g0)
        
    def calcA_g(self, g):
        """
        The wham equation which we aim to minimize,
        A(g1,..,gs): R^s -> R, reference pdf for full expression

        Returns
        -------
        A(g1,...,gs), scaler value

        """
        
        #initialize first two summations
        sum1 = 0
        sum2 = 0
        
        #first term
        for i in self.biasI:
            sum1 = sum1 + (self.N_i * g[i])
        
        #second term
        for l in self.binI:
            if(self.hist[l] != 0):
                #initialize third sum
                sum3 = 0
                for j in self.biasI:
                    sum3 = sum3 + (self.N_i * self.c_i_l[j,l] * np.exp(g[j]))
                    
                if(sum3 != 0):
                    sum2 = sum2 + (self.hist[l] * np.log(self.hist[l]/sum3))
        
        A_g = - sum1 - sum2
        
        return A_g
    
    
    def gradient_dA_dgi(self, g):
        """
        

        Parameters
        ----------
        g : float
            indicies i (1,S)

        Returns
        -------
        dA/dg_i : fload
            indicies i (1/S)

        """
        #initialize gradient vector
        da_dgi = np.zeros(np.size(self.biasI))
        
        for i in self.biasI:
            sum1 = 0
            
            for l in self.binI:
                
                if(self.hist[l] != 0):
                    sum2 = 0
                    
                    for j in self.biasI:
                        sum2 = sum2 + (self.N_i * self.c_i_l[j,l] *
                                       np.exp(g[j]))
                    
                    sum1 = sum1 + ((self.hist[l] * self.c_i_l[i,l])/sum2)
            
            da_dgi[i] = self.N_i * ((np.exp(g[i]) * sum1) - 1)
            
        return da_dgi
                    
                    
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
# %% Test cell

xmin = 0
xmax = 3
bStepNum = 60
N = 10000000
k = 16
binSize = 0.001
fname = 'data-files/xmn0_xmx3_bStepNum60_binSize0.001_k16_N10000000_smoothFunc.txt'

test = WhamConvergence_1Ion(xmin, xmax, bStepNum, N, binSize, k, fname)

(p0,f0,g0) = test.initialEstimates()
        
print(g0) 

A_g0 = test.calcA_g(g0)
print(A_g0)

dA_dgi0 = test.gradient_dA_dgi(g0)
print(dA_dgi0)
        
        
        
        
        
        