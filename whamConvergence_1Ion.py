"""
author: Cameron Kopp

created: 5/15/2024
"""
#IMPORTS
import numpy as np
import scipy.integrate as integrate
from numpy import random as r
from math import *
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
        self.binX = (self.binI * self.binSize) + (self.binSize/2)
        
        #set up two arrays, coordinates of the center of biasing potential
        #for each step biasX, and the indicies for the biasing steps baisI
        stepSize = (self.xmx-self.xmn)/(self.bStepNum + 1)
        self.biasI = np.arange(0, self.bStepNum, dtype=int)
        self.biasX = (self.biasI * stepSize) + stepSize
        
        #histogram array
        self.hist = np.zeros(np.size(self.binI))
        
        
        #read histogram data from file
        with open(self.fileName) as myFile:
            st= myFile.read();
            
        lst = st.split()
        for i in self.binI:
            self.hist[i] = float(lst[i])
            
        # make data
        x = self.binI
        y = self.hist
         
        # plot
        fig, ax = plt.subplots()
         
        ax.plot(x, y, linewidth=2.0)
         
        plt.show()
        
        
        
        
        
        
        
# %% Test cell

xmin = 0
xmax = 3
bStepNum = 60
N = 10000000
k = 16
binSize = 0.001
fname = 'data-files/xmn0_xmx3_bStepNum60_binSize0.001_k16_N10000000_smoothFunc.txt'

test = WhamConvergence_1Ion(xmin, xmax, bStepNum, N, binSize, k, fname)
        
        
        
        
        
        
        
        