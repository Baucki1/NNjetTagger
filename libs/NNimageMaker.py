from __future__ import print_function
from __future__ import division

import ROOT
import numpy as np
import calosampling

#import root_numpy
#from root_numpy import hist2array


class CNNimageMaker():
    def __init__(self, eta=[15, -0.3, 0.3], phi=[15, -np.pi/2, np.pi/2], use_root_histos=True, root_numpy_available=False, verbose=0) -> None:
        self.eta = eta
        self.phi = phi
        self.Sampling = calosampling.CaloSampling()
        
        self.use_root_histos = use_root_histos
        self.root_numpy_available = False
        
        self.verbose = verbose
        
        if use_root_histos and root_numpy_available:
            try:
                import root_numpy
                from root_numpy import hist2array
                self.root_numpy_available = True
            except ImportError:
                self.root_numpy_available = False
                print("root_numpy not available, using numpy instead")    
            
        if self.use_root_histos:
            self.imDict = self.__MakeDictionary(self.eta, self.phi)
        else:
            self.imDict = self.__MakeDictionary_np(self.eta, self.phi)
        
        if self.verbose > 0: print("CNNimageMakerTool loaded")
        return

    def __TH3F_to_np(self, th3) -> np.array:
        return np.array([[[th3.GetBinContent(i,j,k) for k in range(1,th3.GetNbinsZ()+1)] for j in range(1,th3.GetNbinsY()+1)] for i in range(1,th3.GetNbinsX()+1)])

    def __reset_histos(self) -> None:
        if self.use_root_histos:
            for x in ["bar","end","ext"]:
                self.imDict[x].Reset()
        else:
            for x in ["bar","end","ext"]:
                self.imDict[x] = np.zeros((self.eta[0], self.phi[0], self.imDict[x].shape[2]))
        return
        
    def __MakeDictionary(self, eta, phi) -> dict:
        ### Creating input objects for images
        image3D_bar = ROOT.TH3F("jet_3D_bar", "", eta[0], eta[1], eta[2], phi[0], phi[1], phi[2], 4, 0, 4)
        image3D_end = ROOT.TH3F("jet_3D_end", "", eta[0], eta[1], eta[2], phi[0], phi[1], phi[2], 5, 0, 5)
        image3D_ext = ROOT.TH3F("jet_3D_ext", "", eta[0], eta[1], eta[2], phi[0], phi[1], phi[2], 3, 0, 3)
        image3D_dict = {"bar" : image3D_bar, "end" : image3D_end, "ext" : image3D_ext}
        return image3D_dict
    
    def __MakeDictionary_np(self, eta, phi) -> dict:
        return {"bar" : np.zeros((eta[0], phi[0], 4)), "end" : np.zeros((eta[0], phi[0], 5)), "ext" : np.zeros((eta[0], phi[0], 3))}

    def MakeImage(self, T, pj) -> list or int:
        """
        Create 3D image of jet from ROOT Tree.

        Args:
            T (ROOT.TTree, ROOT.TChain, TreeHolder): TTree, TChain or TreeHolder object containing the data
            pj (int): index of the jet in the TTree at the current event to be processed

        Returns:
            list or int : In case of no error returns list of 3 np.arrays (barrel, endcap, extended) with the images of the jet. 
        """
        #isType2Matched = False
        #for PJ in range(T.types.size()):
        #    if T.types.at(PJ) == 2:
        #        if T.jet_index.at(pj) == PJ:
        #            isType2Matched = True

        #if not isType2Matched:
        #    return -9999.

        if not (T.jet_gapRatio.at(pj) > 0.9 and T.jet_width.at(pj) >= 0):# and (not T.jet_IsBIB.at(pj))):
            return -999.

        # if no calocluster are saved go to next
        if T.jet_clusEta.at(pj).size() == 0:
            return -990.

        #Find max entry and skip if bad clusters
        skip = False
        bestE = -100
        bestEIndex = 0

        for cluster in range(T.jet_clusEta.at(pj).size()):
            currentE = 0
            for i in range(T.jet_clusESampl.at(pj).at(cluster).size()):
                currentE += T.jet_clusESampl.at(pj).at(cluster).at(i)
            if (T.jet_clusEta.at(pj).at(cluster) == -99999):
                skip = True
                break
            if(currentE >= bestE):
                bestE = currentE
                bestEIndex = cluster            
        if skip:
            return -900.
                
        
        for cluster in range(T.jet_clusEta.at(pj).size()):
            for samplingIndexItr in range(T.jet_clusSamplIndex.at(pj).at(cluster).size()):
                caloSamplingIndex = T.jet_clusSamplIndex.at(pj).at(cluster).at(samplingIndexItr)
                if self.Sampling.index2Name(caloSamplingIndex) not in self.Sampling.allowedLayers: # next if CaloSampling not included in images
                    continue
                clusterEta = T.jet_clusEta.at(pj).at(cluster) - T.jet_clusEta.at(pj).at(bestEIndex)
                clusterPhi = (T.jet_clusPhi.at(pj).at(cluster) - T.jet_clusPhi.at(pj).at(bestEIndex)) % np.pi - np.pi/2
                clusterLayer = self.Sampling.name2Layer(self.Sampling.index2Name(caloSamplingIndex))
                clusterE = T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr)
                if self.use_root_histos: 
                    self.imDict[self.Sampling.index2Region(caloSamplingIndex)].Fill(clusterEta, clusterPhi, clusterLayer, clusterE)
                else:
                    pos_eta = int((clusterEta-self.eta[1])/((self.eta[2]-self.eta[1])/self.eta[0]))
                    pos_phi = int((clusterPhi-self.phi[1])/((self.phi[2]-self.phi[1])/self.phi[0]))
                    if pos_eta < 0 or pos_eta >= self.eta[0] or pos_phi < 0 or pos_phi >= self.phi[0]:
                        continue
                    self.imDict[self.Sampling.index2Region(caloSamplingIndex)][pos_eta, pos_phi, clusterLayer] += clusterE
            
        if self.use_root_histos:
            if self.root_numpy_available:      
                image_bar = np.expand_dims(hist2array(self.imDict["bar"]), axis = 0)
                image_end = np.expand_dims(hist2array(self.imDict["end"]), axis = 0)
                image_ext = np.expand_dims(hist2array(self.imDict["ext"]), axis = 0)
            else:
                image_bar = np.expand_dims(self.__TH3F_to_np(self.imDict["bar"]), axis = 0)
                image_end = np.expand_dims(self.__TH3F_to_np(self.imDict["end"]), axis = 0)
                image_ext = np.expand_dims(self.__TH3F_to_np(self.imDict["ext"]), axis = 0)
        else:
            image_bar = np.expand_dims(self.imDict["bar"], axis = 0)
            image_end = np.expand_dims(self.imDict["end"], axis = 0)
            image_ext = np.expand_dims(self.imDict["ext"], axis = 0)
             
        self.__reset_histos()
        
        return [image_bar, image_end, image_ext]
    
    
    
class GNNgraphMaker():
    def __init__(self) -> None:
        self.sampling = calosampling.CaloSampling()
        
        self.clusESampl_min = 400 #MeV    
    
    
    def MakeGraph(self, T, pj) -> list:
        """
        Create graph of jet from ROOT Tree.

        Args:
            T (ROOT.TTree, ROOT.TChain, TreeHolder): TTree, TChain or TreeHolder object containing the data
            pj (int): index of the jet in the TTree at the current event to be processed

        Returns:
            list : list of 3 np.arrays (barrel, endcap, extended) with the images of the jet. 
        """

        if not (T.jet_gapRatio.at(pj) > 0.9 and T.jet_width.at(pj) >= 0):
            return -999.

        # if no calocluster are saved go to next
        if T.jet_clusEta.at(pj).size() == 0:
            return -990.
        
        #Find max entry and skip if bad clusters  
        skip = False
        bestE = 100 #fix a threshold of 100 MeV on clusters (-100 before)
        bestEIndex = 0

        #loop over clusters
        for cluster in range(T.jet_clusEta.at(pj).size()):
            currentE = 0
            for i in range(T.jet_clusESampl.at(pj).at(cluster).size()):
                currentE += T.jet_clusESampl.at(pj).at(cluster).at(i)
            if (T.jet_clusEta.at(pj).at(cluster) == -99999):
                skip = True
                break
            if(currentE >= bestE):
                bestE = currentE
                bestEIndex = cluster            
        if skip:
            return -900.

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
        
        ubar, vbar, etabar, phibar, layerbar, energybar, energybarabs = [ubar, ubar, ubar, ubar], [vbar, vbar, vbar, vbar], [etabar, etabar, etabar, etabar], [phibar, phibar, phibar, phibar], [layerbar, layerbar, layerbar, layerbar], [energybar, energybar, energybar, energybar], [energybarabs, energybarabs, energybarabs, energybarabs]
        
        
        energybar0 = []
        energybarabs0 = []
        clusterEbar0 = 0
        clusterEbarabs0 = 0
        #energyend7 = []
        #clusterEend7 = 0
        for cluster in range(T.jet_clusEta.at(pj).size()):
            
            clusterEbar = 0
            clusterEbarabs = 0
            
            for samplingIndexItr in range(T.jet_clusSamplIndex.at(pj).at(cluster).size()):
                
                caloSamplingIndex = T.jet_clusSamplIndex.at(pj).at(cluster).at(samplingIndexItr)
                if self.sampling.index2Name(caloSamplingIndex) not in self.sampling.allowedLayers: # next if CaloSampling not included in images
                    continue
                if (T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr)) < self.clusESampl_min:
                    continue
                clusterLayer = self.sampling.name2Layer(self.sampling.index2Name(caloSamplingIndex))
                clusterEta = T.jet_clusEta.at(pj).at(cluster) - T.jet_clusEta.at(pj).at(bestEIndex)
                clusterPhi = T.jet_clusPhi.at(pj).at(cluster) - T.jet_clusPhi.at(pj).at(bestEIndex)
                
                #Sum the energy of all EM activity in the barrel for each cluster in the Sampling Layer 0 (for all iterations "samplingIndexItr") 
                if clusterLayer == 0:
                    clusterEbar0 = (T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr))*Escala
                    clusterEbarabs0 = (T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr))
                    #print("current E0: ", clusterEbar0)
                    energybar0.append((T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr))*Escala)
                    energybarabs0.append((T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr)))
                    #print("vector energy 0: ", energybar0)
                elif clusterLayer >0 and clusterLayer < 4:
                    #Initialise/Reset cluster energy (1 input) and fill in vectors for all other Sampling layers in Barrel (one node for each/iteration "samplingIndexItr")
                    clusterEbar = (T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr))*Escala
                    clusterEbarabs = (T.jet_clusESampl.at(pj).at(cluster).at(samplingIndexItr))
                    #print("energy in layer (1:TileBar0), (2:TileBar1) or (3:TileBar2)    ", clusterEbar)
                    vbar[clusterLayer].append([clusterEta, clusterPhi])
                    ubar[clusterLayer].append([clusterEta, clusterPhi, clusterEbar])
                    energybar[clusterLayer].append([clusterEbar])
                    energybarabs[clusterLayer].append([clusterEbarabs])
                    etabar[clusterLayer].append([clusterEta])
                    phibar[clusterLayer].append([clusterPhi])
                    layerbar[clusterLayer].append([clusterLayer])
                    
                    for i in range(0,4):
                        if i == clusterLayer: continue
                        vbar[i].append([epsilon, epsilon])
                        ubar[i].append([epsilon, epsilon, epsilon])
                        energybar[i].append([epsilon])
                        energybarabs[i].append([epsilon])
                        etabar[i].append([epsilon])
                        phibar[i].append([epsilon])
                        layerbar[i].append([i])
            
            
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
                
                vbar[0].append([clusterEta, clusterPhi])
                ubar[0].append([clusterEta, clusterPhi, clusterEbar0])
                energybar[0].append([clusterEbar0])
                energybarabs[0].append([clusterEbarabs0])
                etabar[0].append([clusterEta])
                phibar[0].append([clusterPhi])
                layerbar[0].append([clusterLayer])
                
                for i in range(1,4):
                    vbar[i].append([epsilon, epsilon])
                    ubar[i].append([epsilon, epsilon, epsilon])
                    energybar[i].append([epsilon])
                    energybarabs[i].append([epsilon])
                    etabar[i].append([epsilon])
                    phibar[i].append([epsilon])
                    layerbar[i].append([i])
                
                
                #ubar.append([clusterEta+epsilon, clusterPhi+epsilon, clusterLayer+epsilon])
                nodesbar += 1
                energybar0 = []
                energybarabs0 = []
                clusterEbar0 = 0
                clusterEbarabs0 = 0
                
        keysbar = np.arange(0, nodesbar)
        
        nodes_posbar, nodes_posbar2, nodes_etabar, nodes_phibar, nodes_layerbar, nodes_energybar, nodes_energybarabs = [], [], [], [], [], [], []
        for i in range(4):
            nodes_posbar.append(dict(zip(keysbar, vbar[i])))
            #nodes_CNNscores.append({key: T.jet_DPJtagger.at(pj) for key in keysbar})
            nodes_posbar2.append(dict(zip(keysbar, ubar[i])))
            nodes_etabar.append(dict(zip(keysbar, etabar[i])))
            nodes_phibar.append(dict(zip(keysbar, phibar[i])))
            nodes_layerbar.append(dict(zip(keysbar, layerbar[i])))
            nodes_energybar.append(dict(zip(keysbar, energybar[i])))
            nodes_energybarabs.append(dict(zip(keysbar, energybarabs[i])))
        
        return [nodes_posbar, nodes_posbar2, nodes_etabar, nodes_phibar, nodes_layerbar, nodes_energybar, nodes_energybarabs]