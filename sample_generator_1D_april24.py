#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:59:55 2022

@author: cameronkopp
"""

from numpy import *
from numpy import random as r
from math import *
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import colors

#SAMPLE CONSTRUCTOR CLASS ===================================================

class SampleConstructor1D:
    """
        inputs: xmin, xmax (such that the sample will span range xmin-xmax),
        simNumber (number of simulations to run over x coordinate range, will be
                equally spaced), 
        binSize (distance spanned by each bin over x coordinate, smaller 
                 binSize means more precision and a larger file),
        springConstant (wieghted MD simulations commonly refer to the biasing
                        potential coefficient as the spring constant,
                        here we use it to calculate the standard deviation of
                        our gaussian),
        sampleNumber (number of samples taken from each simulation)
    """
    
    def __init__(self, xmin, xmax, simNumber, binSize, springConstant, 
                 sampleNumber):
        
        """
        Class Constructor: initalizes paramaters and sets up the necessary
        grid for the histogram.
        """
        #Read in parameters
        self.xmn = xmin
        self.xmx = xmax
        self.sNum = simNumber
        self.bs = binSize
        self.K = springConstant
        self.N = sampleNumber
        
        #Set up bin grid, bc = coordinate of the start of each bin
        # bi is an integer index corresponding the bc array
        self.bc = arange(self.xmn, self.xmx, self.bs)
        self.bi = arange(0, ((self.xmx-self.xmn)/self.bs), dtype=int)
        
        #Set up coordinates for center of biasing potential, simc is the 
        # coordinate for the center of the biasing potential, simi is an
        # integer index corresponding to the simc array
        simstep = (self.xmx - self.xmn)/self.sNum
        self.simc = arange(self.xmn, self.xmx, simstep)
        self.simi = arange(0, size(self.simc), dtype=int)
        
        #initialize the histogram as an array of zeros matching bc array
        self.hist = zeros(size(self.bi))
        
    #Method for computing the biasing potential for a given spring constant
    def biasV(self, x, mu):
        """
        Defining the biasing potential ahead in interest of readable code,
        for most weighted MD simulations it is just the harmonic oscillator
        which gives a probability distribution of a gaussian with a standard
        deviation defined by the spring constant,
        x is going to be the coordinate of the simulated particle,
        mu is the coordinate of the center of the biasing potential
        """
        #DEFINE CONSTANTS
       
        c1 = self.K/2
        c2 = sqrt(c1/math.pi)
        
        #contribution to probability distribution from the biasing potential
        pV = c2*exp(-c1*(x-mu)**2)
        
        return pV
    
    #Method for known probablity distribution
    def xSinSq(self, x):
        """
        first probabilty distribution tested, rho(x)=c*x*sin^2(x), 
        need to add a constant c to ensure that the value is never > 1
        """
        
        #normalization constant
        c = 1/self.xmx
        rho = c*x*(sin(x)**2)
        
        return rho
    
    def xp1nverseSinSq(self, x):
        """
        test distribution that could work a bit better
        """

        rho = (1/(x+1))*(sin(x)**2)
        return rho
    
    def xp1nverseSinSq2x(self, x):
        """
        test distribution that could work a bit better
        """

        rho = (1/(x+1))*(sin(2*x)**2)
        return rho
    
    def newFunc(self, x):
        """
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        f(x) = (1/(1+((1-x)**2)))*((sin(2*x)**2)+2)/3
        """
        rho = (1/(1+(1-x)**2))*((sin(2*x)**2)+2)/3
        return rho
    
    #generate the histogram using xSinSq distribution
    def dataGen1D(self):
        """
        Relying on an Acceptance-Rejection algorithm to generate the data,
        for details read pdf

        """
        
        xtrange = 4/sqrt(self.K)
        fmax = 1
        #each simulation will be centered around an element of the
        #array simc, called mu after gaussian convention
        for mu in self.simc:
            # calculate a range of values from mu 4 stdvs away from gaussian
            
            print(mu)
            i = 0        #accepted sample counter = i
            
            #loop over accept-reject while i < sampleNumber
            while(i < self.N):
                #test point
                xt = r.uniform(mu - xtrange, mu + xtrange)
                #biased prob dist of test point
                rhoxt = self.biasV(xt,mu)*self.xp1nverseSinSq(xt) 
                #random 0-fmax
                y = r.uniform(0,fmax)
                #if the prob dist of the test point is greater than
                #random 0-fmax, accept the point
                if rhoxt > y:
                    #accept sample even if it is not in xmn-xmx
                    # i = i + 1
                    if (i == floor(self.N/2)):
                        print('1/2 way')
                    if (xt < self.xmx) and (xt > self.xmn):
                        i = i + 1
                        #add accepted point to the correct bin
                        #convert from float to int
                        histbinf = (xt/self.bs)
                        histbin = floor(histbinf)
                        #hi is the histogram index that xt corresponds to
                        hi = int(histbin)
                        self.hist[hi] = self.hist[hi] + 1
            
        figure()
        plot(self.bc, self.hist)
        xlabel('x')
        ylabel('histogram(x)')
        title('')
        show()
        
        return self.hist
    
    def dataGen1D_v2(self):
        """
        Relying on an Acceptance-Rejection algorithm to generate the data,
        for details read pdf, 
        _v2: utilized the fact that (the biasing potential) *
        (the probability distribution) will always be less than the biasing
        potential alone, seeing as the prob dist is defined as less than one,
        improves speed drastically

        """
        
        
        stdev = 1/sqrt(self.K)
        #each simulation will be centered around an element of the
        #array simc, called mu after gaussian convention
        for mu in self.simc:
            # calculate a range of values from mu 4 stdvs away from gaussian
            
            print(mu)
            i = 0        #accepted sample counter = i
            
            #loop over accept-reject while i < sampleNumber
            while(i < self.N):
                #test point
                xt = r.normal(mu,stdev)
                #biased prob dist of test point
                rhoxt = self.biasV(xt,mu)*self.newFunc(xt)
                #random 0-fmax
                y = r.uniform(0,self.biasV(xt, mu))
                #if the prob dist of the test point is greater than
                #random 0-fmax, accept the point
                if rhoxt > y:
                    #accept sample even if it is not in xmn-xmx
                    # i = i + 1
                    # if (i == floor(self.N/2)):
                    #     print('1/2 way')
                    if (xt < self.xmx) and (xt > self.xmn):
                        #add accepted point to the correct bin
                        #convert from float to int
                        histbinf = (xt/self.bs)
                        histbin = floor(histbinf)
                        #hi is the histogram index that xt corresponds to
                        hi = int(histbin)
                        self.hist[hi] = self.hist[hi] + 1
                        i = i + 1
                        if (i == floor(self.N/2)):
                            print('1/2 way')
            
        figure()
        plot(self.bc, self.hist)
        xlabel('x')
        ylabel('histogram(x)')
        title('')
        show()
        
        return self.hist
    
    #funnction to write histogram data to txt file
    def writeToFile(self, funcUsed):
        """
        formats a file name from class parameters and writes histogram
        data to a txt file in same folder, funcUsed is the name of the 
        distribution function used to generate the data

        """
        filename = "data-files/xmn{}_xmx{}_simNum{}_bs{}_k{}_n{}_{}.txt".format(self.xmn,
                    self.xmx, self.sNum, self.bs, self.K, self.N, funcUsed)
        
        with open(filename,'w') as myfile:
            for x in self.bi:
                myfile.write('{} '.format(self.hist[x]))
                
        myfile.close()
        
        return
                    
                
                
                
            
            
    
    
        
        
        
        
        
        
        
# %% 
#TESTING CELL

xmn = 0
xmx = 3
simnum = 120
binsize = 1/500
spK = 4
sampnum = 100000
sampnum = 100000

testGen = SampleConstructor1D(xmn, xmx, simnum, binsize, spK, sampnum)
firsthist = testGen.dataGen1D_v2()
testGen.writeToFile('newFunc')


        