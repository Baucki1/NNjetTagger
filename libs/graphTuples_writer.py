#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:13:03 2022

@author: carmigna
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import os
import ROOT
import calosampling
import argparse
import sys
import numpy as np
#from sklearn import preprocessing
import deepdish as dd

def getOptions(args=sys.argv[1:]):
    ###  Argument parser ###
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i","--input", nargs="+", help="Input files, expects either comma-separated .root files or .txt list of files", type=str)
    parser.add_argument("-o","--outputDir", nargs="+", help="Output dir", type=str)
    
    options = parser.parse_args(args)
    return options

class TreeHolder():
    def __init__(self,T):
        self.TTree = T
    def upd(self):
        self.types = self.TTree.types
        self.LJjet_index = self.TTree.LJjet_index 
        self.LJjet_IsBIB = self.TTree.LJjet_IsBIB       
        self.GNNjet_gapRatio = self.TTree.GNNjet_gapRatio
        self.GNNjet_timing = self.TTree.GNNjet_timing
        self.GNNjet_jvt = self.TTree.GNNjet_jvt
        self.GNNjet_eta = self.TTree.GNNjet_eta
        self.GNNjet_phi = self.TTree.GNNjet_phi
        self.GNNjet_m = self.TTree.GNNjet_m
        self.GNNjet_width = self.TTree.GNNjet_width
        self.GNNjet_clusEta = self.TTree.GNNjet_clusEta
        self.GNNjet_clusPhi = self.TTree.GNNjet_clusPhi
        self.GNNjet_clusESampl = self.TTree.GNNjet_clusESampl
        self.GNNjet_clusSamplIndex = self.TTree.GNNjet_clusSamplIndex
        self.GNNjet_DPJtagger = self.TTree.GNNjet_DPJtagger
#GNNjet_DPJtagger









#######################
###     MAIN       ####
#######################

if __name__ == "__main__":
    options = getOptions(sys.argv[1:])
    Sampling = calosampling.CaloSampling()
    ### Collect input files
    inputFilesList = []
    print("\n\n\nRetrieving input files:")
    if options.input:
        for ntupleFile in options.input:
            if ntupleFile.endswith(".txt"):
                lines = [line.rstrip('\n') for line in open(ntupleFile)]
                for line in lines:
                    if line=="" : continue
                    if line.endswith(".root"):
                        inputFilesList.append(line)
                        print(line)
                    else:
                        raise Exception(line+" does not end with .root")
            elif os.path.isdir(ntupleFile):
                for ntupleFileItr in os.listdir(ntupleFile):
                    if ntupleFileItr.endswith(".root"):
                        inputFilesList.append(os.path.join(ntupleFile,ntupleFileItr))
                        print(os.path.join(ntupleFile,ntupleFileItr))
            elif ntupleFile.endswith(".root"):
                if ("," in ntupleFile ):
                    for ntupleFileItr in ntupleFile.split(","):
                        if ntupleFileItr.endswith(".root"):
                            inputFilesList.append(ntupleFileItr)
                            print(ntupleFileItr)
                        else:
                            raise Exception(ntupleFile+" does not end with .root")
                else: ### ntupleFile is a single file
                    if ntupleFile.endswith(".root"):
                        inputFilesList.append(ntupleFile)
                        print(ntupleFile)
            else:
                raise Exception("Input file syntax can not be understood: please fix\n"+ntupleFile)

    print("\n * * * Creating TChain with input files * * * \n")
    T = ROOT.TChain("miniT")
    for x in inputFilesList:
        T.Add(x)
        print(" > adding file to TChain: ",x)

    print("\n\n\n * * * graphMLWriter running * * *\n\n\n")
    
    ### Flags
    tagAllLJ = True ## if false evaluate NN only for leading & far
    reader = TreeHolder(T)
    ### Removing unwanted branches
    #branchStatus = [["trigsys*",0],
                    #["LJjet_clus*",0],
                    #["extra*",0],
                    #["*Lead*LJ",0],
                    #["numCostituentsLJ",0],
                    #["trackMultiplicityID",0],
                    #["LJmuon_*lay*",0],
                    #["HLT*",0]]

    #for tag, status in branchStatus:
    #    T.SetBranchStatus(tag,status)

    ### Enable caloclusters from old ntuple
    T.SetBranchStatus("GNNjet_clus*",1)

    ###############################
    ###    LOOP over entries   ####
    ###############################
    nEntries = T.GetEntries()
    nEntriesFilled = 0     
    print("Total entries in TChain:", nEntries)
    print(" * * * Begin ntuple loop * * * ")
    sys.stdout.flush()
    tj = 0
    
    barnodes = {}

    CNNscores = {}
    barpos3 = {}
    barpos4 = {}
    bareta = {}
    barphi = {}
    barlayer = {}
    barenergy = {}
    barenergyabs = {}
    
    #a1_barnodes = {}
    a1_CNNscores = {}
    a1_barpos3 = {}
    a1_barpos4 = {}
    a1_bareta = {}
    a1_barphi = {}
    a1_barlayer = {}
    a1_barenergy = {}
    a1_barenergyabs = {}
    
    a2_CNNscores = {}
    a2_barpos3 = {}
    a2_barpos4 = {}
    a2_bareta = {}
    a2_barphi = {}
    a2_barlayer = {}
    a2_barenergy = {}
    a2_barenergyabs = {}
    
    #a3_barnodes = {}
    a3_CNNscores = {}
    a3_barpos3 = {}
    a3_barpos4 = {}
    a3_bareta = {}
    a3_barphi = {}
    a3_barlayer = {}
    a3_barenergy = {}
    a3_barenergyabs = {}
# =============================================================================
#     extnodes = {}
#     extpos3 = {}
#     extpos4 = {}
#     exteta = {}
#     extphi = {}
#     extlayer = {}
#     extenergy = {}
#     endnodes = {}
#     endpos3 = {}
#     endpos4 = {}
#     endeta = {}
#     endphi = {}
#     endlayer = {}
#     endenergy = {}
# =============================================================================
    for entry in range(150000):#nEntries):#for signal nEntries instead of #
        T.GetEntry(entry)        
        reader.upd()                
        if entry > 0 and entry%10000==0:
            print("Processed {} of {} entries".format(entry,150000))#nEntries))#
        lead_index = T.lead_index
        far_index = T.far20_index
            
        if tj < 150000:#nEntries:#
            # GNNjet loop for "Barrel" region N_sig limit for Jet/Graphs (N_sig/2 + N_sig/2 for background)
            for lj in range(T.GNNjet_eta.size()):
                # General cuts not needed with Skimmed GNN data
                #if ( tagAllLJ or ( T.LJjet_index[lj] == lead_index ) or ( T.LJjet_index[lj] == far_index ) ):
                    #isType2Matched = False                
                    #for LJ in range(T.types.size()):
                    #    if T.types.at(LJ) == 2:
                    #        if T.LJjet_index.at(lj) == LJ:
                    #            isType2Matched = True
                    #if not isType2Matched:
                    #    continue
                    #if not (T.GNNjet_gapRatio.at(lj) > 0.9 and T.GNNjet_width.at(lj) >= 0): # and (not T.LJjet_IsBIB.at(lj))):
                    #    continue
                    #if no calocluster are saved go to next
    
                #A cut on the absolute value of the jets to stay in the barrel
                if abs(T.GNNjet_eta.at(lj)) > 1.0 :
                    continue 
                if T.GNNjet_clusEta.at(lj).size() == 0:
                    continue
    
    
                #Find max entry and skip if bad clusters  
                skip = False
                bestE = 100#fix a threshold of 100 MeV on clusters (-100 before)
                bestEIndex = 0
    
                #loop over clusters
                for cluster in range(T.GNNjet_clusEta.at(lj).size()):
                    currentE = 0
                    for i in range(T.GNNjet_clusESampl.at(lj).at(cluster).size()):
                        currentE += T.GNNjet_clusESampl.at(lj).at(cluster).at(i)
                    if (T.GNNjet_clusEta.at(lj).at(cluster) == -99999):
                        skip = True
                        break
                    if(currentE >= bestE):
                        bestE = currentE
                        bestEIndex = cluster            
                if skip:
                    continue
                tj += 1
    
                #Create a scale to normalise with highest energy
                Escala=1/bestE
                #initialise input vectors then fill
                epsilon = 1e-8
                
                nodesbar = 2
                ubar = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
                vbar = [[epsilon,epsilon], [epsilon,epsilon]]
                etabar = [[epsilon], [epsilon]]
                phibar = [[epsilon], [epsilon]]
                layerbar = [[0], [0]]
                energybar = [[epsilon], [epsilon]]
                energybarabs = [[epsilon], [epsilon]]
                
                #a1_nodesbar = 2
                a1_ubar = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
                a1_vbar = [[epsilon,epsilon], [epsilon,epsilon]]
                a1_etabar = [[epsilon],[epsilon]]
                a1_phibar = [[epsilon],[epsilon]]
                a1_layerbar = [[1],[1]]
                a1_energybar = [[epsilon],[epsilon]]
                a1_energybarabs = [[epsilon],[epsilon]]
                
                #a2_nodesbar = 2
                a2_ubar = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
                a2_vbar = [[epsilon,epsilon], [epsilon,epsilon]]
                a2_etabar = [[epsilon],[epsilon]]
                a2_phibar = [[epsilon],[epsilon]]
                a2_layerbar = [[2],[2]]
                a2_energybar = [[epsilon],[epsilon]]
                a2_energybarabs = [[epsilon],[epsilon]]
                
                #a3_nodesbar = 2
                a3_ubar = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
                a3_vbar = [[epsilon,epsilon], [epsilon,epsilon]]
                a3_etabar = [[epsilon],[epsilon]]
                a3_phibar = [[epsilon],[epsilon]]
                a3_layerbar = [[3],[3]]
                a3_energybar = [[epsilon],[epsilon]]
                a3_energybarabs = [[epsilon],[epsilon]]
                
                
# =============================================================================
#                 nodesend = 2
#                 uext = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
#                 vext = [[epsilon,epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon,epsilon]] 
#                 etaext = [[epsilon], [epsilon]]
#                 phiext = [[epsilon], [epsilon]]
#                 layerext = [[5+epsilon], [5+epsilon]]
#                 energyext = [[epsilon], [epsilon]]
#                 nodesext = 2
#                 uend = [[epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon]] 
#                 vend = [[epsilon,epsilon,epsilon,epsilon], [epsilon,epsilon,epsilon,epsilon]]
#                 etaend = [[epsilon], [epsilon]]
#                 phiend = [[epsilon], [epsilon]]
#                 layerend = [[7+epsilon], [7+epsilon]]
#                 energyend = [[epsilon], [epsilon]]
# =============================================================================
                
                #g = True
                
                energybar0 = []
                energybarabs0 = []
                clusterEbar0 = 0
                clusterEbarabs0 = 0
                #energyend7 = []
                #clusterEend7 = 0
                for cluster in range(T.GNNjet_clusEta.at(lj).size()):
                    #reset energy inputs for Sampling layer 0
                    
                    clusterEbar = 0
                    clusterEbarabs = 0
                    
                    for samplingIndexItr in range(T.GNNjet_clusSamplIndex.at(lj).at(cluster).size()):
                        
                        caloSamplingIndex = T.GNNjet_clusSamplIndex.at(lj).at(cluster).at(samplingIndexItr)
                        # next if CaloSampling not included in images
                        if Sampling.index2Name(caloSamplingIndex) not in Sampling.allowedLayers:
                            continue
                        # Create a cut of 400 MeV
                        if (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr)) < 400:
                            continue
                        clusterLayer = Sampling.name2Layer(Sampling.index2Name(caloSamplingIndex))
                        clusterEta = T.GNNjet_clusEta.at(lj).at(cluster) - T.GNNjet_clusEta.at(lj).at(bestEIndex)
                        clusterPhi = T.GNNjet_clusPhi.at(lj).at(cluster) - T.GNNjet_clusPhi.at(lj).at(bestEIndex)
                        
                        #print(clusterEta)
                        #print(clusterPhi)
                        #print(clusterLayer)
                        
                        
                        #Sum the energy of all EM activity in the barrel for each cluster in the Sampling Layer 0 (for all iterations "samplingIndexItr") 
                        if clusterLayer == 0 :
                            clusterEbar0 = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))*Escala
                            clusterEbarabs0 = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))
                            #print("current E0: ", clusterEbar0)
                            energybar0.append((T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))*Escala)
                            energybarabs0.append((T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr)))
                            #print("vector energy 0: ", energybar0)
# =============================================================================
#                         #Sum the energy of all EM activity in the End-Cap for each cluster in the Sampling Layer 7 (for all iterations "samplingIndexItr")
#                         if clusterLayer == 7:
#                             clusterEend7 = T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr)*Escala
#                             #print("current E7: ", clusterEend7)
#                             energyend7.append(T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))
#                             #print("vector energy 7: ", energyend7)
# =============================================================================
                            
                        #Initialise/Reset cluster energy (1 input) and fill in vectors for all other Sampling layers in Barrel (one node for each/iteration "samplingIndexItr")
                        
                        if clusterLayer == 1:
                            clusterEbar = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))*Escala
                            clusterEbarabs = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))
                            #print("energy in layer (1:TileBar0), (2:TileBar1) or (3:TileBar2)    ", clusterEbar)
                            a1_vbar.append([clusterEta, clusterPhi])
                            a1_ubar.append([clusterEta, clusterPhi, clusterEbar])
                            a1_energybar.append([clusterEbar])
                            a1_energybarabs.append([clusterEbarabs])
                            a1_etabar.append([clusterEta])
                            a1_phibar.append([clusterPhi])
                            a1_layerbar.append([clusterLayer])
                            
                            vbar.append([epsilon, epsilon])
                            ubar.append([epsilon, epsilon, epsilon])
                            energybar.append([epsilon])
                            energybarabs.append([epsilon])
                            etabar.append([epsilon])
                            phibar.append([epsilon])
                            layerbar.append([0])
                            
                            a2_vbar.append([epsilon, epsilon])
                            a2_ubar.append([epsilon, epsilon, epsilon])
                            a2_energybar.append([epsilon])
                            a2_energybarabs.append([epsilon])
                            a2_etabar.append([epsilon])
                            a2_phibar.append([epsilon])
                            a2_layerbar.append([2])
                            
                            a3_vbar.append([epsilon, epsilon])
                            a3_ubar.append([epsilon, epsilon, epsilon])
                            a3_energybar.append([epsilon])
                            a3_energybarabs.append([epsilon])
                            a3_etabar.append([epsilon])
                            a3_phibar.append([epsilon])
                            a3_layerbar.append([3])
                            
                            nodesbar += 1
                            
                        if clusterLayer == 2:
                            clusterEbar = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))*Escala
                            clusterEbarabs = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))
                            #print("energy in layer (1:TileBar0), (2:TileBar1) or (3:TileBar2)    ", clusterEbar)
                            a2_vbar.append([clusterEta, clusterPhi])
                            a2_ubar.append([clusterEta, clusterPhi,clusterEbar])
                            a2_energybar.append([clusterEbar])
                            a2_energybarabs.append([clusterEbarabs])
                            a2_etabar.append([clusterEta])
                            a2_phibar.append([clusterPhi])
                            a2_layerbar.append([clusterLayer])
                            
                            a1_vbar.append([epsilon, epsilon])
                            a1_ubar.append([epsilon, epsilon, epsilon])
                            a1_energybar.append([epsilon])
                            a1_energybarabs.append([epsilon])
                            a1_etabar.append([epsilon])
                            a1_phibar.append([epsilon])
                            a1_layerbar.append([1])
                            
                            vbar.append([epsilon, epsilon])
                            ubar.append([epsilon, epsilon, epsilon])
                            energybar.append([epsilon])
                            energybarabs.append([epsilon])
                            etabar.append([epsilon])
                            phibar.append([epsilon])
                            layerbar.append([0])
                            
                            a3_vbar.append([epsilon, epsilon])
                            a3_ubar.append([epsilon, epsilon, epsilon])
                            a3_energybar.append([epsilon])
                            a3_energybarabs.append([epsilon])
                            a3_etabar.append([epsilon])
                            a3_phibar.append([epsilon])
                            a3_layerbar.append([3])
                            
                            nodesbar += 1
                            
                        if clusterLayer == 3:
                            clusterEbar = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))*Escala
                            clusterEbarabs = (T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr))
                            #print("energy in layer (1:TileBar0), (2:TileBar1) or (3:TileBar2)    ", clusterEbar)
                            a3_vbar.append([clusterEta, clusterPhi])
                            a3_ubar.append([clusterEta, clusterPhi, clusterEbar])
                            a3_energybar.append([clusterEbar])
                            a3_energybarabs.append([clusterEbarabs])
                            a3_etabar.append([clusterEta])
                            a3_phibar.append([clusterPhi])
                            a3_layerbar.append([clusterLayer])
                            
                            a1_vbar.append([epsilon, epsilon])
                            a1_ubar.append([epsilon, epsilon, epsilon])
                            a1_energybar.append([epsilon])
                            a1_energybarabs.append([epsilon])
                            a1_etabar.append([epsilon])
                            a1_phibar.append([epsilon])
                            a1_layerbar.append([1])
                            
                            a2_vbar.append([epsilon, epsilon])
                            a2_ubar.append([epsilon, epsilon, epsilon])
                            a2_energybar.append([epsilon])
                            a2_energybarabs.append([epsilon])
                            a2_etabar.append([epsilon])
                            a2_phibar.append([epsilon])
                            a2_layerbar.append([2])
                            
                            vbar.append([epsilon, epsilon])
                            ubar.append([epsilon, epsilon, epsilon])
                            energybar.append([epsilon])
                            energybarabs.append([epsilon])
                            etabar.append([epsilon])
                            phibar.append([epsilon])
                            layerbar.append([0])
                            
                            nodesbar += 1
                            
                            
                            
                            
                            
                            
                            
                        
                        #Initialise/Reset cluster energy (1 input) and fill in vectors for all Sampling layers in Extended Region (one node for each/iteration "samplingIndexItr")
                        
# =============================================================================
#                         if (clusterLayer == 4) or (clusterLayer == 5) or (clusterLayer == 6):
#                             clusterEext = T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr)*Escala
#                             #print("energy in layer (4:TileExt0), (5:TileExt1) or (6:TileExt2)    ", clusterEext)
#                             vext.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon, clusterEext+epsilon])
#                             energyext.append([clusterEext+epsilon])
#                             etaext.append([clusterEta+epsilon])
#                             phiext.append([clusterPhi+epsilon])
#                             layerext.append([clusterLayer+epsilon])
#                             uext.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon])
#                             nodesext += 1
#                             #print("nodes in Extended: ", nodesext)               
#                             #print("v in Extended: ", vext)  
#                             #print("eta in Extended: ", etaext)
#                             #print("phi in Extended: ", phiext)
#                             #print("layer in Extended: ", layerext)
#                             #print("energy in Extended: ", energyext)
#                         
#                         #Initialise/Reset cluster energy (1 input) and fill in vectors for all other Sampling layers in End-Cap (one node for each/iteration "samplingIndexItr")
#                         
#                         if (clusterLayer == 8) or (clusterLayer == 9) or (clusterLayer == 10) or (clusterLayer == 11):
#                             clusterEend = T.GNNjet_clusESampl.at(lj).at(cluster).at(samplingIndexItr)*Escala
#                             #print("energy in layer (8:HEC0), (9:HEC1), (10:HEC2) or (11:HEC3)   ", clusterEend)
#                             vend.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon, clusterEend+epsilon])
#                             energyend.append([clusterEend+epsilon])
#                             etaend.append([clusterEta+epsilon])
#                             phiend.append([clusterPhi+epsilon])
#                             layerend.append([clusterLayer+epsilon])
#                             uend.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon])
#                             nodesend += 1
#                             #print("nodes in EndCap not 7: ", nodesend)               
#                             #print("v in EndCap not 7: ", vend)  
#                             #print("eta in EndCap not 7: ", etaend)
#                             #print("phi in EndCap not 7: ", phiend)
#                             #print("layer in EndCap not 7: ", layerend)
#                             #print("energy in EndCap not 7: ", energyend)
# =============================================================================
                        
                    
                    if (clusterEbar0 != 0):
                        #Fill in vectors for total EM energy in Sampling Layer 0 for each cluster (build one node/cluster)
                        clusterEbar0 = sum(energybar0)
                        clusterEbarabs0 = sum(energybarabs0)
                        #print("sum energy in layer 0 (Barrel EM): ", clusterEbar0)
                        vbar.append([clusterEta, clusterPhi])
                        ubar.append([clusterEta, clusterPhi, clusterEbar0])
                        energybar.append([clusterEbar0])
                        energybarabs.append([clusterEbarabs0])
                        etabar.append([clusterEta])
                        phibar.append([clusterPhi])
                        layerbar.append([clusterLayer])
                        
                        a1_vbar.append([epsilon, epsilon])
                        a1_ubar.append([epsilon, epsilon, epsilon])
                        a1_energybar.append([epsilon])
                        a1_energybarabs.append([epsilon])
                        a1_etabar.append([epsilon])
                        a1_phibar.append([epsilon])
                        a1_layerbar.append([1])
                        
                        a2_vbar.append([epsilon, epsilon])
                        a2_ubar.append([epsilon, epsilon, epsilon])
                        a2_energybar.append([epsilon])
                        a2_energybarabs.append([epsilon])
                        a2_etabar.append([epsilon])
                        a2_phibar.append([epsilon])
                        a2_layerbar.append([2])
                        
                        a3_vbar.append([epsilon, epsilon])
                        a3_ubar.append([epsilon, epsilon, epsilon])
                        a3_energybar.append([epsilon])
                        a3_energybarabs.append([epsilon])
                        a3_etabar.append([epsilon])
                        a3_phibar.append([epsilon])
                        a3_layerbar.append([3])
                        
                        
                        #ubar.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon])
                        nodesbar += 1
                        energybar0 = []
                        energybarabs0 = []
                        clusterEbar0 = 0
                        clusterEbarabs0 = 0
    # =============================================================================
    #                     print("nodes in Barrel at 0: ", nodesbar)               
    #                     print("v in Barrel at 0: ", vbar)  
    #                     print("eta in Barrel at 0: ", etabar)
    #                     print("phi in Barrel at 0: ", phibar)
    #                     print("layer in Barrel at 0: ", layerbar)
    #                     print("energy in Barrel at 0: ", energybar)
    # =============================================================================
                        
# =============================================================================
#                     if (clusterEend7 != 0):
#                         #Fill in vectors for total EM energy in Sampling Layer 0 for each cluster (build one node/cluster)
#                         clusterEend7 = sum(energyend7)
#                         #print("sum energy in layer 7 (End-Cap EM): ", clusterEend7)
#                         vend.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon, clusterEend7+epsilon])
#                         energyend.append([clusterEend7+epsilon])
#                         etaend.append([clusterEta+epsilon])
#                         phiend.append([clusterPhi+epsilon])
#                         layerend.append([clusterLayer+epsilon])
#                         uend.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon])
#                         nodesend += 1
#     # =============================================================================
#     #                     print("nodes in EndCap at 7: ", nodesend)               
#     #                     print("v in EndCap at 7: ", vend)  
#     #                     print("eta in EndCap at 7: ", etaend)
#     #                     print("phi in EndCap at 7: ", phiend)
#     #                     print("layer in EndCap at 7: ", layerend)
#     #                     print("energy in EndCap at 7: ", energyend)
#     # =============================================================================
#                         energyend7 = []
#                         clusterEend7 = 0
# =============================================================================
                    
# =============================================================================
#                 #Normalise the vectors in Barrel
#                 nUbar = ubar
#                 nVbar = ((preprocessing.normalize((np.array(vbar)).T)).T).tolist()
#                 nEtabar = ((preprocessing.normalize((np.array(etabar)).T)).T).tolist()
#                 nPhibar = ((preprocessing.normalize((np.array(phibar)).T)).T).tolist()
#                 nLayerbar = ((preprocessing.normalize((np.array(layerbar)).T)).T).tolist()
#                 nEnergybar = energybar 
# =============================================================================
    # =============================================================================
    #             print("Final  3-vector bar Normalised:  ", nUbar)
    #             print("Final  4-vector bar Normalised:  ", nVbar)
    #             print("Final Eta bar Normalised:  ", nEtabar)
    #             print("Final Phi bar Normalised:  ", nPhibar)
    #             print("Final Layer bar Normalised:  ", nLayerbar)
    #             print("Final E bar Normalised:  ", nEnergybar)
    # =============================================================================
                    
    
# =============================================================================
#                 #Normalise the vectors in End-Cap
#                 nUend = uend
#                 nVend = ((preprocessing.normalize((np.array(vend)).T)).T).tolist()
#                 nEtaend = ((preprocessing.normalize((np.array(etaend)).T)).T).tolist()
#                 nPhiend = ((preprocessing.normalize((np.array(phiend)).T)).T).tolist()
#                 nLayerend = ((preprocessing.normalize((np.array(layerend)).T)).T).tolist()
#                 nEnergyend = energyend
#     # =============================================================================
#     #             print("Final  3-vector end Normalised:  ", nUend)
#     #             print("Final  4-vector end Normalised:  ", nVend)
#     #             print("Final  Eta end Normalised:  ", nEtaend)
#     #             print("Final  Phi end Normalised:  ", nPhiend)
#     #             print("Final  Layer end Normalised:  ", nLayerend)
#     #             print("Final  E end Normalised:  ", nEnergyend)
#     # =============================================================================
#                     
#      
#                 #Normalise the vectors in Extended Barrel
#                 nUext = uext
#                 nVext = ((preprocessing.normalize((np.array(vext)).T)).T).tolist()
#                 nEtaext = ((preprocessing.normalize((np.array(etaext)).T)).T).tolist()
#                 nPhiext = ((preprocessing.normalize((np.array(phiext)).T)).T).tolist()
#                 nLayerext = ((preprocessing.normalize((np.array(layerext)).T)).T).tolist()
#                 nEnergyext = energyext 
# =============================================================================
                
                
                
    #           #add keys 
                keysbar = np.arange(0,nodesbar)
                #keysbar1 = np.arange(nodesbar,nodesbar*2)
                #keysbar2 = np.arange(nodesbar*2,nodesbar*3)
                #keysbar3 = np.arange(nodesbar*3,nodesbar*4)
    #           #print("keys barrel: ",keysbar)
                #keysext = np.arange(0,nodesext)
    #           #print("keys extended: ",keysext)
                #keysend = np.arange(0,nodesend)
    #           #print("keys endcap: ",keysend)
    # =============================================================================
    #             print("Final  3-vector ext Normalised:  ", nUext)
    #             print("Final  4-vector ext Normalised:  ", nVext)
    #             print("Final  Eta ext Normalised:  ", nEtaext)
    #             print("Final  Phi ext Normalised:  ", nPhiext)
    #             print("Final  Layer ext Normalised:  ", nLayerext)
    #             print("Final  E ext Normalised:  ", nEnergyext)
    # =============================================================================
                    
                
    
    
    
    
    
                ##############################
                ###Building Networkx Graphs###
                ##############################
    
    
    # =============================================================================
    
    #     
    #             ###Define "Hertz Distributions" PDF of metric-based connections from "self-arranging NNDD" for random paticles in taxicab geometries### 
    #             #References:
    #             #(Phys. Rev. Lett. 41(4)1990)
    #             # Chandrasekhar, S. (1943-01-01). "Stochastic Problems in Physics and Astronomy". Reviews of Modern Physics. 15 (1): 1–89. Bibcode:1943RvMP...15....1C. doi:10.1103/RevModPhys.15.1
    #             # Hertz, Paul (1909). "Über den gegenseitigen durchschnittlichen Abstand von Punkten, die mit bekannter mittlerer Dichte im Raume angeordnet sind". Mathematische Annalen. 67 (3): 387–398. 
    #             # doi:10.1007/BF01450410. ISSN 0025-5831. S2CID 120573104. 
    # 
    #             #initial density particles/site
    #             ro = 1# 100% 
    #  
    #             def p_dist(r):
    #                 return ro*(4*r+2)*math.exp(ro*(-2*(r**2)-2*r-1))#for Hertz PDF #/r ###for linear distribution
    # 
    #             
    #             
    #             #Define threshold "theta" (Radius of connection)
    #             theta = 20#(moderate computation value)
    #             
    #             #add edge attributes or ["measure"] (scaled "Manhattan taxicab" metric)
    #             scale = epsilon
    #             #Define "Manhattan" taxicab metric for grids  (https://en.wikipedia.org/wiki/Taxicab_geometry)
    #             distt = lambda x, y: sum(abs(a - b)  for a, b in zip(x, y))
    #             
    # =============================================================================
                
                #Write geographical graph 
                #draws edges above threshold
                #Jgraph = nx.geographical_threshold_graph(n=nodes,theta=theta,pos=nodes_pos,metric=dist, p_dist=p_dist)
                
                
                
                nodes_posbar = dict(zip(keysbar, vbar))
                nodes_CNNscores = {key: T.GNNjet_DPJtagger.at(lj) for key in keysbar}
                nodes_posbar2 = dict(zip(keysbar, ubar))
                nodes_etabar = dict(zip(keysbar, etabar))
                nodes_phibar = dict(zip(keysbar, phibar))
                nodes_layerbar = dict(zip(keysbar, layerbar))
                nodes_energybar = dict(zip(keysbar, energybar))
                nodes_energybarabs = dict(zip(keysbar, energybarabs))
                
                barnodes[tj] = nodesbar
                CNNscores[tj] = nodes_CNNscores
                barpos3[tj] = nodes_posbar
                barpos4[tj] = nodes_posbar2
                bareta[tj] = nodes_etabar
                barphi[tj] = nodes_phibar
                barlayer[tj] = nodes_layerbar
                barenergy[tj] = nodes_energybar
                barenergyabs[tj] = nodes_energybarabs
                
                a1_nodes_posbar = dict(zip(keysbar, a1_vbar))
                a1_nodes_CNNscores = {key: T.GNNjet_DPJtagger.at(lj) for key in keysbar}
                a1_nodes_posbar2 = dict(zip(keysbar, a1_ubar))
                a1_nodes_etabar = dict(zip(keysbar, a1_etabar))
                a1_nodes_phibar = dict(zip(keysbar, a1_phibar))
                a1_nodes_layerbar = dict(zip(keysbar, a1_layerbar))
                a1_nodes_energybar = dict(zip(keysbar, a1_energybar))
                a1_nodes_energybarabs = dict(zip(keysbar, a1_energybarabs))
                
                #a1_barnodes[tj] = a1_nodesbar
                a1_CNNscores[tj] = a1_nodes_CNNscores
                a1_barpos3[tj] = a1_nodes_posbar
                a1_barpos4[tj] = a1_nodes_posbar2
                a1_bareta[tj] = a1_nodes_etabar
                a1_barphi[tj] = a1_nodes_phibar
                a1_barlayer[tj] = a1_nodes_layerbar
                a1_barenergy[tj] = a1_nodes_energybar
                a1_barenergyabs[tj] = a1_nodes_energybarabs
                
                a2_nodes_posbar = dict(zip(keysbar, a2_vbar))
                a2_nodes_CNNscores = {key: T.GNNjet_DPJtagger.at(lj) for key in keysbar}
                a2_nodes_posbar2 = dict(zip(keysbar, a2_ubar))
                a2_nodes_etabar = dict(zip(keysbar, a2_etabar))
                a2_nodes_phibar = dict(zip(keysbar, a2_phibar))
                a2_nodes_layerbar = dict(zip(keysbar, a2_layerbar))
                a2_nodes_energybar = dict(zip(keysbar, a2_energybar))
                a2_nodes_energybarabs = dict(zip(keysbar, a2_energybarabs))
                
                #a2_barnodes[tj] = a2_nodesbar
                a2_CNNscores[tj] = a2_nodes_CNNscores
                a2_barpos3[tj] = a2_nodes_posbar
                a2_barpos4[tj] = a2_nodes_posbar2
                a2_bareta[tj] = a2_nodes_etabar
                a2_barphi[tj] = a2_nodes_phibar
                a2_barlayer[tj] = a2_nodes_layerbar
                a2_barenergy[tj] = a2_nodes_energybar
                a2_barenergyabs[tj] = a2_nodes_energybarabs
                
                a3_nodes_posbar = dict(zip(keysbar, a3_vbar))
                a3_nodes_CNNscores = {key: T.GNNjet_DPJtagger.at(lj) for key in keysbar}
                a3_nodes_posbar2 = dict(zip(keysbar, a3_ubar))
                a3_nodes_etabar = dict(zip(keysbar, a3_etabar))
                a3_nodes_phibar = dict(zip(keysbar, a3_phibar))
                a3_nodes_layerbar = dict(zip(keysbar, a3_layerbar))
                a3_nodes_energybar = dict(zip(keysbar, a3_energybar))
                a3_nodes_energybarabs = dict(zip(keysbar, a3_energybarabs))
                
                #a3_barnodes[tj] = a3_nodesbar
                a3_CNNscores[tj] = a3_nodes_CNNscores
                a3_barpos3[tj] = a3_nodes_posbar
                a3_barpos4[tj] = a3_nodes_posbar2
                a3_bareta[tj] = a3_nodes_etabar
                a3_barphi[tj] = a3_nodes_phibar
                a3_barlayer[tj] = a3_nodes_layerbar
                a3_barenergy[tj] = a3_nodes_energybar
                a3_barenergyabs[tj] = a3_nodes_energybarabs
                nEntriesFilled += 1

   ### End of entries loop

    if  nEntriesFilled != 150000:#nEntries:#
        #print("Something has gone wrong...")
        print("nEntries:          ", 150000)#nEntries)#
        print("tj iters:          ", tj)
        print("nEntriesFilled:", nEntriesFilled)
        #print("Exit with ERROR!")
        #exit(1)

    ### Save and close
    outfilebarnodes = options.outputDir[0] + ("a0_tupleGraph_barnodes.h5") 
    dd.io.save(outfilebarnodes, barnodes)
    outfileCNNscores = options.outputDir[0] + ("a0_tupleGraph_barCNNscores.h5") 
    dd.io.save(outfileCNNscores, CNNscores)
    outfilebarpos3 = options.outputDir[0] + ("a0_tupleGraph_barpos3.h5") 
    dd.io.save(outfilebarpos3, barpos3)
    outfilebarpos4 = options.outputDir[0] + ("a0_tupleGraph_barpos4.h5") 
    dd.io.save(outfilebarpos4, barpos4)
    outfilebareta = options.outputDir[0] + ("a0_tupleGraph_bareta.h5") 
    dd.io.save(outfilebareta, bareta)
    outfilebarphi = options.outputDir[0] + ("a0_tupleGraph_barphi.h5") 
    dd.io.save(outfilebarphi, barphi)
    outfilebarlayer = options.outputDir[0] + ("a0_tupleGraph_barlayer.h5") 
    dd.io.save(outfilebarlayer, barlayer)
    outfilebarenergy = options.outputDir[0] + ("a0_tupleGraph_barenergy.h5") 
    dd.io.save(outfilebarenergy, barenergy)
    outfilebarenergyabs = options.outputDir[0] + ("a0_tupleGraph_barenergyabs.h5") 
    dd.io.save(outfilebarenergyabs, barenergyabs)
    
    #a1_outfilebarnodes = options.outputDir[0] + ("a1_b3tupleGraph_barnodes.h5") 
    #dd.io.save(a1_outfilebarnodes, a1_barnodes)
    a1_outfileCNNscores = options.outputDir[0] + ("a1_tupleGraph_barCNNscores.h5") 
    dd.io.save(a1_outfileCNNscores, a1_CNNscores)
    a1_outfilebarpos3 = options.outputDir[0] + ("a1_tupleGraph_barpos3.h5") 
    dd.io.save(a1_outfilebarpos3, a1_barpos3)
    a1_outfilebarpos4 = options.outputDir[0] + ("a1_tupleGraph_barpos4.h5") 
    dd.io.save(a1_outfilebarpos4, a1_barpos4)
    a1_outfilebareta = options.outputDir[0] + ("a1_tupleGraph_bareta.h5") 
    dd.io.save(a1_outfilebareta, a1_bareta)
    a1_outfilebarphi = options.outputDir[0] + ("a1_tupleGraph_barphi.h5") 
    dd.io.save(a1_outfilebarphi, a1_barphi)
    a1_outfilebarlayer = options.outputDir[0] + ("a1_tupleGraph_barlayer.h5") 
    dd.io.save(a1_outfilebarlayer, a1_barlayer)
    a1_outfilebarenergy = options.outputDir[0] + ("a1_tupleGraph_barenergy.h5") 
    dd.io.save(a1_outfilebarenergy, a1_barenergy)
    a1_outfilebarenergyabs = options.outputDir[0] + ("a1_tupleGraph_barenergyabs.h5") 
    dd.io.save(a1_outfilebarenergyabs, a1_barenergyabs)
    
    #a2_outfilebarnodes = options.outputDir[0] + ("a2_b3tupleGraph_barnodes.h5") 
    #dd.io.save(a2_outfilebarnodes, a2_barnodes)
    a2_outfileCNNscores = options.outputDir[0] + ("a2_tupleGraph_barCNNscores.h5") 
    dd.io.save(a2_outfileCNNscores, a2_CNNscores)
    a2_outfilebarpos3 = options.outputDir[0] + ("a2_tupleGraph_barpos3.h5") 
    dd.io.save(a2_outfilebarpos3, a2_barpos3)
    a2_outfilebarpos4 = options.outputDir[0] + ("a2_tupleGraph_barpos4.h5") 
    dd.io.save(a2_outfilebarpos4, a2_barpos4)
    a2_outfilebareta = options.outputDir[0] + ("a2_tupleGraph_bareta.h5") 
    dd.io.save(a2_outfilebareta, a2_bareta)
    a2_outfilebarphi = options.outputDir[0] + ("a2_tupleGraph_barphi.h5") 
    dd.io.save(a2_outfilebarphi, a2_barphi)
    a2_outfilebarlayer = options.outputDir[0] + ("a2_tupleGraph_barlayer.h5") 
    dd.io.save(a2_outfilebarlayer, a2_barlayer)
    a2_outfilebarenergy = options.outputDir[0] + ("a2_tupleGraph_barenergy.h5") 
    dd.io.save(a2_outfilebarenergy, a2_barenergy)
    a2_outfilebarenergyabs = options.outputDir[0] + ("a2_tupleGraph_barenergyabs.h5") 
    dd.io.save(a2_outfilebarenergyabs, a2_barenergyabs)
    
    #a3_outfilebarnodes = options.outputDir[0] + ("a3_b3tupleGraph_barnodes.h5") 
    #dd.io.save(a3_outfilebarnodes, a3_barnodes)
    a3_outfileCNNscores = options.outputDir[0] + ("a3_tupleGraph_barCNNscores.h5") 
    dd.io.save(a3_outfileCNNscores, a3_CNNscores)
    a3_outfilebarpos3 = options.outputDir[0] + ("a3_tupleGraph_barpos3.h5") 
    dd.io.save(a3_outfilebarpos3, a3_barpos3)
    a3_outfilebarpos4 = options.outputDir[0] + ("a3_tupleGraph_barpos4.h5") 
    dd.io.save(a3_outfilebarpos4, a3_barpos4)
    a3_outfilebareta = options.outputDir[0] + ("a3_tupleGraph_bareta.h5") 
    dd.io.save(a3_outfilebareta, a3_bareta)
    a3_outfilebarphi = options.outputDir[0] + ("a3_tupleGraph_barphi.h5") 
    dd.io.save(a3_outfilebarphi, a3_barphi)
    a3_outfilebarlayer = options.outputDir[0] + ("a3_tupleGraph_barlayer.h5") 
    dd.io.save(a3_outfilebarlayer, a3_barlayer)
    a3_outfilebarenergy = options.outputDir[0] + ("a3_tupleGraph_barenergy.h5") 
    dd.io.save(a3_outfilebarenergy, a3_barenergy)
    a3_outfilebarenergyabs = options.outputDir[0] + ("a3_tupleGraph_barenergyabs.h5") 
    dd.io.save(a3_outfilebarenergyabs, a3_barenergyabs)
    

    
    ROOT.gDirectory.Delete("T;*")

    sys.exit(0)


        
    
