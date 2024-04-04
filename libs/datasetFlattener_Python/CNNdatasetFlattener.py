from array import array

import numpy as np

import ROOT
from ROOT import TFile, TTree, TBranch, std

# run set_event_branches and set_jet_branches before flatten 
# (e.g. 
# datasetFlattener.set_event_branches("default") 
# datasetFlattener.set_jet_branches("default")
# datasetFlattener.flatten() )
class datasetFlattener:
    def __init__(self, input_file, output_file, tree_name="T"):
        """Flatten a TTree to one jet per event

        Args:
            input_file (str): path to input file
            output_file (_type_): path to output file
            tree_name (str, optional): name of tree. Defaults to "T".
        """
        
        self.input_file = input_file
        self.output_file = output_file
        self.tree_name = tree_name

        self.__event_branches = []
        self.__jet_branches = []
        self.__default_event_branches = ["isMC", "RunNumber", "dsid", "eventNumber", "amiXsection", "filterEff", "kFactor", "sumWeightPRW", "lead_index", "far20_index", "etaLJ", "phiLJ", "ptLJ", "types", "isoID", "LJ_index", "LJ_MatchedTruthDPindex", "weight", "puWeight", "mcWeight", "truthPdgId", "truthEta", "truthPhi", "truthPt", "truthE", "truthCharge", "truthBarcode", "truthDecayVtx_x", "truthDecayVtx_y", "truthDecayVtx_z", "truthDecayType", "childPdgId", "childEta", "childPhi", "childPt", "childBarcode", "childMomBarcode"]
        self.__default_jet_branches = ["GNNjet_eta", "GNNjet_phi", "GNNjet_pt", "GNNjet_width", "GNNjet_EMfrac", "GNNjet_timing", "GNNjet_jvt", "GNNjet_gapRatio", "GNNjet_m", "GNNjet_clusESampl", "GNNjet_clusSamplIndex", "GNNjet_clusEta", "GNNjet_clusPhi"]
        
        
    def set_event_branches(self, event_branches): # set "global" quantities (one per event)
        """Set which branches to use for event-level quantities

        Args:
            event_branches (list or str): list of branch names or "default" for default list
        """
        
        if type(event_branches) is list:
            self.__event_branches = event_branches
        elif type(event_branches) is str and event_branches == "default":
            self.__event_branches = self.__default_event_branches
        else:
            print("Event branches not supported: ", event_branches)
        
        input_file = TFile(self.input_file, "READ")
        input_tree = input_file.Get(self.tree_name)
        branch_list = input_tree.GetListOfBranches()
        #input_file.close()
        
        for branch in self.__event_branches:
            if branch not in branch_list:
                print("Branch not found in input tree: ", branch, " - removing from event_branches")
                self.__event_branches.remove(branch)
        return


    def set_jet_branches(self, jet_branches): # set jet-specific quantities (one per jet)
        """Set which branches to use for jet-level quantities

        Args:
            jet_branches (list or str): list of branch names or "default" for default list
        """
        
        if type(jet_branches) is list:
            self.jet_branches = jet_branches
        elif type(jet_branches) is str and jet_branches == "default":
            self.__jet_branches = self.__default_jet_branches
        else:
            print("Jet branches not supported: ", jet_branches)
            
        input_file = TFile(self.input_file, "READ")
        input_tree = input_file.Get(self.tree_name)
        branch_list = input_tree.GetListOfBranches()
        #input_file.close()
        
        for branch in self.__jet_branches:
            if branch not in branch_list:
                print("Branch not found in input tree: ", branch, " - removing from jet_branches")
                self.__jet_branches.remove(branch)
                continue
                
            #if input_tree.GetBranch(branch).GetLeaf(branch).GetLen() == 1:
            #    print("Branch is not a vector: ", branch, " - removing from jet_branches")
            #    self.__jet_branches.remove(branch)
        return


    def flatten(self):
        """Flatten the input tree to one jet per event (iterate over branches of jet quantities (jet_branches) and copy event quantities (event_branches) for each jet in the event)

        Returns:
            str: path to output file.
        """
        
        if len(self.__event_branches) == 0:
            print("No event branches set")
            return
        
        if len(self.__jet_branches) == 0:
            print("No jet branches set")
            return
        
        # Open the input file
        input_root_file = TFile(self.input_file, "READ")
        input_tree = input_root_file.Get(self.tree_name)

        # Create the output file
        output_root_file = TFile(self.output_file, "RECREATE")
        output_tree = ROOT.TTree(self.tree_name, self.tree_name)
        
        # Create the output branches
        output_branches = {}
        for branch_title in self.__event_branches:
            branch = input_tree.GetBranch(branch_title)
            branch_name = branch.GetName()
            branch_type = branch.GetListOfLeaves().At(0).GetTypeName()
            if "F" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.float32)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/F")
            elif "I" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.int32)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/I")
            elif "O" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.bool)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/O")
            elif "vector<float>" in branch_type:
                output_branches[branch_name] = std.vector("float")([0])
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name)
            elif "vector<int>" in branch_type:
                output_branches[branch_name] = std.vector("int")([0])
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name)
            else:
                print("Branch type not supported: ", branch_type) 
                print("Branch not added: ", branch_name)
                self.__event_branches.remove(branch_title)
            
        for branch_title in self.__jet_branches:
            branch = input_tree.GetBranch(branch_title)
            branch_name = branch.GetName()
            branch_type = branch.GetListOfLeaves().At(0).GetTypeName()
            if "F" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.float32)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/F")
            elif "I" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.int32)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/I")
            elif "O" in branch_type:
                output_branches[branch_name] = np.zeros(1, dtype=np.bool)
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name + "/O")
            elif "vector<float>" in branch_type:
                output_branches[branch_name] = std.vector("float")([0])
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name)
            elif "vector<int>" in branch_type:
                output_branches[branch_name] = std.vector("int")([0])
                output_tree.Branch(branch_name, output_branches[branch_name], branch_name)
            #elif "vector<vector<float>>" in branch_type:
            #elif "vector<vector<int>>" in branch_type:
            #elif "vector<vector<vector<float>>" in branch_type:
            else:
                print("Branch type not supported: ", branch_type)
                print("Branch not added: ", branch_name)
                self.__jet_branches.remove(branch_title)
            
        
        print("Flattening tree: ", self.tree_name, "(entries: ", input_tree.GetEntries(), ")")
        print("Event branches used: ", self.__event_branches)
        print("Jet branches used: ", self.__jet_branches)
        
        # Fill the output tree with a new entry for each jet (=> 1jet/event)
        for entry in range(input_tree.GetEntries()):
            input_tree.GetEntry(entry)
            n_jets = input_tree.GNNjet_width.size() #input_tree.GetLeaf(self.__jet_branches[0]).GetLen()
            for jet in range(n_jets):
                for branch_name in self.__event_branches:
                    branch = input_tree.GetBranch(branch_name)
                    #output_branches[branch_name][0] = branch.GetLeaf(branch_name).GetValue()
                    if type(output_branches[branch_name]) == np.ndarray:
                        output_branches[branch_name][0] = branch.GetLeaf(branch_name).GetValue()
                    elif type(output_branches[branch_name]) == std.vector("float"):
                        output_branches[branch_name].clear()
                        for i in range(branch.GetLeaf(branch_name).GetLen()):
                            output_branches[branch_name].push_back(branch.GetLeaf(branch_name).GetValue(i))
                    #elif type(output_branches[branch_name]) == std.vector("int"): # gives a TypeError??
                        #output_branches[branch_name].clear()
                        #for i in range(branch.GetLeaf(branch_name).GetLen()):
                            #if entry == 0: print(branch.GetLeaf(branch_name).GetValue(i))
                            #output_branches[branch_name].push_back(branch.GetLeaf(branch_name).GetValue(i))
                
                for branch_name in self.__jet_branches:
                    branch = input_tree.GetBranch(branch_name)
                    #output_branches[branch_name][0] = branch.GetLeaf(branch_name).GetValue(jet)
                    if type(output_branches[branch_name]) == np.ndarray:
                        output_branches[branch_name][0] = branch.GetLeaf(branch_name).GetValue(jet)
                    elif type(output_branches[branch_name]) == std.vector("float"):
                        output_branches[branch_name].clear()
                        for i in range(branch.GetLeaf(branch_name).GetLen()):
                            output_branches[branch_name].push_back(branch.GetLeaf(branch_name).GetValue(i))
                    #elif type(output_branches[branch_name]) == std.vector("int"): # gives a TypeError??
                        #output_branches[branch_name].clear()
                        #for i in range(branch.GetLeaf(branch_name).GetLen()):
                            #if entry == 0: print(branch.GetLeaf(branch_name).GetValue(i))
                            #output_branches[branch_name].push_back(branch.GetLeaf(branch_name).GetValue(i))
                    
                output_tree.Fill()
                
            if entry%(input_tree.GetEntries()/100) == 0:
                print("entry", str(entry) + '/' + str(input_tree.GetEntries()), "("+str(round(entry/input_tree.GetEntries()*100, 2))+"%)", end='\r')
        
        # Write the output file
        output_root_file.Write()
        output_root_file.Close()
        input_root_file.Close()
        
        print("File created: ", self.output_file)
        return self.output_file
    
    
    
    
# (not working) alternative implementation explicitly setting the branches
class datasetFlattener_uglypieceofshit:
    def __init__(self, input_file, output_file, tree_name="T"):
        self.input_file = input_file
        self.output_file = output_file
        self.tree_name = tree_name        
    
    def flatten(self):
        # Open the input file
        input_root_file = TFile(self.input_file, "READ")
        input_tree = input_root_file.Get(self.tree_name)

        # Create the output file
        output_root_file = TFile(self.output_file, "RECREATE")
        output_tree = ROOT.TTree(self.tree_name, self.tree_name)
        
        # Create the output branches
        isMC = output_tree.Branch("isMC", np.zeros(1, dtype=np.bool), "isMC/O")
        RunNumber = output_tree.Branch("RunNumber", np.zeros(1, dtype=np.int32), "RunNumber/I")
        dsid = output_tree.Branch("dsid", np.zeros(1, dtype=np.int32), "dsid/I")
        eventNumber = output_tree.Branch("eventNumber", np.zeros(1, dtype=np.int32), "eventNumber/I")
        amiXsection = output_tree.Branch("amiXsection", np.zeros(1, dtype=np.float32), "amiXsection/F")
        filterEff = output_tree.Branch("filterEff", np.zeros(1, dtype=np.float32), "filterEff/F")
        kFactor = output_tree.Branch("kFactor", np.zeros(1, dtype=np.float32), "kFactor/F")
        sumWeightPRW = output_tree.Branch("sumWeightPRW", np.zeros(1, dtype=np.float32), "sumWeightPRW/F")
        lead_index = output_tree.Branch("lead_index", np.zeros(1, dtype=np.int32), "lead_index/I")
        far20_index = output_tree.Branch("far20_index", np.zeros(1, dtype=np.int32), "far20_index/I")
        etaLJ = output_tree.Branch("etaLJ", std.vector("float")([0]), "etaLJ")
        phiLJ = output_tree.Branch("phiLJ", std.vector("float")([0]), "phiLJ")
        ptLJ = output_tree.Branch("ptLJ", std.vector("float")([0]), "ptLJ")
        types = output_tree.Branch("types", std.vector("int")([0]), "types")
        isoID = output_tree.Branch("isoID", std.vector("float")([0]), "isoID")
        LJ_index = output_tree.Branch("LJ_index", std.vector("int")([0]), "LJ_index")
        LJ_MatchedTruthDPindex = output_tree.Branch("LJ_MatchedTruthDPindex", std.vector(std.vector("int"))([0][0]), "LJ_MatchedTruthDPindex")
        weight = output_tree.Branch("weight", np.zeros(1, dtype=np.float32), "weight/F")
        puWeight = output_tree.Branch("puWeight", np.zeros(1, dtype=np.float32), "puWeight/F")
        mcWeight = output_tree.Branch("mcWeight", np.zeros(1, dtype=np.int32), "mcWeight/I")
        LJjet_index = output_tree.Branch("LJjet_index", std.vector("int")([0]), "LJjet_index")
        LJjet_eta = output_tree.Branch("LJjet_eta", std.vector("float")([0]), "LJjet_eta")
        LJjet_phi = output_tree.Branch("LJjet_phi", std.vector("float")([0]), "LJjet_phi")
        LJjet_pt = output_tree.Branch("LJjet_pt", std.vector("float")([0]), "LJjet_pt")
        LJjet_width = output_tree.Branch("LJjet_width", std.vector("float")([0]), "LJjet_width")
        LJjet_EMfrac = output_tree.Branch("LJjet_EMfrac", std.vector("float")([0]), "LJjet_EMfrac")
        LJjet_timing = output_tree.Branch("LJjet_timing", std.vector("float")([0]), "LJjet_timing")
        LJjet_jvt = output_tree.Branch("LJjet_jvt", std.vector("float")([0]), "LJjet_jvt")
        LJjet_gapRatio = output_tree.Branch("LJjet_gapRatio", std.vector("float")([0]), "LJjet_gapRatio")
        LJjet_IsBIB = output_tree.Branch("LJjet_IsBIB", std.vector("float")([0]), "LJjet_IsBIB")
        LJjet_m = output_tree.Branch("LJjet_m", std.vector("float")([0]), "LJjet_m")
        truthPdgId = output_tree.Branch("truthPdgId", std.vector("int")([0]), "truthPdgId")
        truthEta = output_tree.Branch("truthEta", std.vector("float")([0]), "truthEta")
        truthPhi = output_tree.Branch("truthPhi", std.vector("float")([0]), "truthPhi")
        truthPt = output_tree.Branch("truthPt", std.vector("float")([0]), "truthPt")
        truthE = output_tree.Branch("truthE", std.vector("float")([0]), "truthE")
        truthCharge = output_tree.Branch("truthCharge", std.vector("float")([0]), "truthCharge")
        truthBarcode = output_tree.Branch("truthBarcode", std.vector("int")([0]), "truthBarcode")
        truthDecayVtx_x = output_tree.Branch("truthDecayVtx_x", std.vector("float")([0]), "truthDecayVtx_x")
        truthDecayVtx_y = output_tree.Branch("truthDecayVtx_y", std.vector("float")([0]), "truthDecayVtx_y")
        truthDecayVtx_z = output_tree.Branch("truthDecayVtx_z", std.vector("float")([0]), "truthDecayVtx_z")
        truthDecayType = output_tree.Branch("truthDecayType", std.vector("int")([0]), "truthDecayType")
        childPdgId = output_tree.Branch("childPdgId", std.vector("int")([0]), "childPdgId")
        childEta = output_tree.Branch("childEta", std.vector("float")([0]), "childEta")
        childPhi = output_tree.Branch("childPhi", std.vector("float")([0]), "childPhi")
        childPt = output_tree.Branch("childPt", std.vector("float")([0]), "childPt")
        childBarcode = output_tree.Branch("childBarcode", std.vector("int")([0]), "childBarcode")
        childMomBarcode = output_tree.Branch("childMomBarcode", std.vector("int")([0]), "childMomBarcode")
        GNNjet_eta = output_tree.Branch("GNNjet_eta", std.vector("float")([0]), "GNNjet_eta")
        GNNjet_phi = output_tree.Branch("GNNjet_phi", std.vector("float")([0]), "GNNjet_phi")
        GNNjet_pt = output_tree.Branch("GNNjet_pt", std.vector("float")([0]), "GNNjet_pt")
        GNNjet_width = output_tree.Branch("GNNjet_width", std.vector("float")([0]), "GNNjet_width")
        GNNjet_EMfrac = output_tree.Branch("GNNjet_EMfrac", std.vector("float")([0]), "GNNjet_EMfrac")
        GNNjet_timing = output_tree.Branch("GNNjet_timing", std.vector("float")([0]), "GNNjet_timing")
        GNNjet_jvt = output_tree.Branch("GNNjet_jvt", std.vector("float")([0]), "GNNjet_jvt")
        GNNjet_gapRatio = output_tree.Branch("GNNjet_gapRatio", std.vector("float")([0]), "GNNjet_gapRatio")
        GNNjet_m = output_tree.Branch("GNNjet_m", std.vector("float")([0]), "GNNjet_m")
        GNNjet_clusESampl = output_tree.Branch("GNNjet_clusESampl", std.vector(std.vector(std.vector("float"))([0][0][0])), "GNNjet_clusESampl")
        GNNjet_clusSamplIndex = output_tree.Branch("GNNjet_clusSamplIndex", std.vector(std.vector(std.vector("int"))([0][0][0])), "GNNjet_clusSamplIndex")
        GNNjet_clusEta = output_tree.Branch("GNNjet_clusEta", std.vector(std.vector("float")([0][0])), "GNNjet_clusEta")
        GNNjet_clusPhi = output_tree.Branch("GNNjet_clusPhi", std.vector(std.vector("float")([0][0])), "GNNjet_clusPhi")
            
        
        for entry in range(input_tree.GetEntries()):
            input_tree.GetEntry(entry)
            for jet in range(input_tree.GNNjet_width.size()):
                isMC.Fill(input_tree.isMC)
                RunNumber.Fill(input_tree.RunNumber)
                dsid.Fill(input_tree.dsid)
                eventNumber.Fill(input_tree.eventNumber)
                amiXsection.Fill(input_tree.amiXsection)
                filterEff.Fill(input_tree.filterEff)
                kFactor.Fill(input_tree.kFactor)
                sumWeightPRW.Fill(input_tree.sumWeightPRW)
                lead_index.Fill(input_tree.lead_index)
                far20_index.Fill(input_tree.far20_index)
                etaLJ.clear()
                for i in range(input_tree.etaLJ.size()):
                    etaLJ.push_back(input_tree.etaLJ[i])
                phiLJ.clear()
                for i in range(input_tree.phiLJ.size()):
                    phiLJ.push_back(input_tree.phiLJ[i])
                ptLJ.clear()
                for i in range(input_tree.ptLJ.size()):
                    ptLJ.push_back(input_tree.ptLJ[i])
                types.clear()
                for i in range(input_tree.types.size()):
                    types.push_back(input_tree.types[i])
                isoID.clear()
                for i in range(input_tree.isoID.size()):
                    isoID.push_back(input_tree.isoID[i])
                LJ_index.clear()
                for i in range(input_tree.LJ_index.size()):
                    LJ_index.push_back(input_tree.LJ_index[i])
                LJ_MatchedTruthDPindex.clear()
                for i in range(input_tree.LJ_MatchedTruthDPindex.size()):
                    LJ_MatchedTruthDPindex.push_back(input_tree.LJ_MatchedTruthDPindex[i])
                weight.Fill(input_tree.weight)
                puWeight.Fill(input_tree.puWeight)
                mcWeight.Fill(input_tree.mcWeight)
                LJjet_index.clear()
                #for i in range(input_tree.LJjet_index.size()):
                LJjet_index.push_back(input_tree.LJjet_index[jet]) #
                LJjet_eta.clear()
                for i in range(input_tree.LJjet_eta.size()):
                    LJjet_eta.push_back(input_tree.LJjet_eta[i])
                LJjet_phi.clear()
                for i in range(input_tree.LJjet_phi.size()):
                    LJjet_phi.push_back(input_tree.LJjet_phi[i])
                LJjet_pt.clear()
                for i in range(input_tree.LJjet_pt.size()):
                    LJjet_pt.push_back(input_tree.LJjet_pt[i])
                LJjet_width.clear()
                for i in range(input_tree.LJjet_width.size()):
                    LJjet_width.push_back(input_tree.LJjet_width[i])
                LJjet_EMfrac.clear()
                for i in range(input_tree.LJjet_EMfrac.size()):
                    LJjet_EMfrac.push_back(input_tree.LJjet_EMfrac[i])
                LJjet_timing.clear()
                for i in range(input_tree.LJjet_timing.size()):
                    LJjet_timing.push_back(input_tree.LJjet_timing[i])
                LJjet_jvt.clear()
                for i in range(input_tree.LJjet_jvt.size()):
                    LJjet_jvt.push_back(input_tree.LJjet_jvt[i])
                LJjet_gapRatio.clear()
                for i in range(input_tree.LJjet_gapRatio.size()):
                    LJjet_gapRatio.push_back(input_tree.LJjet_gapRatio[i])
                LJjet_IsBIB.clear()
                LJjet_IsBIB.push_back(input_tree.LJjet_IsBIB[jet]) #
                LJjet_m.clear()
                for i in range(input_tree.LJjet_m.size()):
                    LJjet_m.push_back(input_tree.LJjet_m[i])
                truthPdgId.clear()
                for i in range(input_tree.truthPdgId.size()):
                    truthPdgId.push_back(input_tree.truthPdgId[i])
                truthEta.clear()
                for i in range(input_tree.truthEta.size()):
                    truthEta.push_back(input_tree.truthEta[i])
                truthPhi.clear()
                for i in range(input_tree.truthPhi.size()):
                    truthPhi.push_back(input_tree.truthPhi[i])
                truthPt.clear()
                for i in range(input_tree.truthPt.size()):
                    truthPt.push_back(input_tree.truthPt[i])
                truthE.clear()
                for i in range(input_tree.truthE.size()):
                    truthE.push_back(input_tree.truthE[i])
                truthCharge.clear()
                for i in range(input_tree.truthCharge.size()):
                    truthCharge.push_back(input_tree.truthCharge[i])
                truthBarcode.clear()
                for i in range(input_tree.truthBarcode.size()):
                    truthBarcode.push_back(input_tree.truthBarcode[i])
                truthDecayVtx_x.clear()
                for i in range(input_tree.truthDecayVtx_x.size()):
                    truthDecayVtx_x.push_back(input_tree.truthDecayVtx_x[i])
                truthDecayVtx_y.clear()
                for i in range(input_tree.truthDecayVtx_y.size()):
                    truthDecayVtx_y.push_back(input_tree.truthDecayVtx_y[i])
                truthDecayVtx_z.clear()
                for i in range(input_tree.truthDecayVtx_z.size()):
                    truthDecayVtx_z.push_back(input_tree.truthDecayVtx_z[i])
                truthDecayType.clear()
                for i in range(input_tree.truthDecayType.size()):
                    truthDecayType.push_back(input_tree.truthDecayType[i])
                childPdgId.clear()
                for i in range(input_tree.childPdgId.size()):
                    childPdgId.push_back(input_tree.childPdgId[i])
                childEta.clear()
                for i in range(input_tree.childEta.size()):
                    childEta.push_back(input_tree.childEta[i])
                childPhi.clear()
                for i in range(input_tree.childPhi.size()):
                    childPhi.push_back(input_tree.childPhi[i])
                childPt.clear()
                for i in range(input_tree.childPt.size()):
                    childPt.push_back(input_tree.childPt[i])
                childBarcode.clear()
                for i in range(input_tree.childBarcode.size()):
                    childBarcode.push_back(input_tree.childBarcode[i])
                childMomBarcode.clear()
                for i in range(input_tree.childMomBarcode.size()):
                    childMomBarcode.push_back(input_tree.childMomBarcode[i])
                GNNjet_eta.clear()
                GNNjet_eta.push_back(input_tree.GNNjet_eta[jet])
                GNNjet_phi.clear()
                GNNjet_phi.push_back(input_tree.GNNjet_phi[jet])
                GNNjet_pt.clear()
                GNNjet_pt.push_back(input_tree.GNNjet_pt[jet])
                GNNjet_width.clear()
                GNNjet_width.push_back(input_tree.GNNjet_width[jet])
                GNNjet_EMfrac.clear()
                GNNjet_EMfrac.push_back(input_tree.GNNjet_EMfrac[jet])
                GNNjet_timing.clear()
                GNNjet_timing.push_back(input_tree.GNNjet_timing[jet])
                GNNjet_jvt.clear()
                GNNjet_jvt.push_back(input_tree.GNNjet_jvt[jet])
                GNNjet_gapRatio.clear()
                GNNjet_gapRatio.push_back(input_tree.GNNjet_gapRatio[jet])
                GNNjet_m.clear()
                GNNjet_m.push_back(input_tree.GNNjet_m[jet])
                GNNjet_clusESampl.clear()
                GNNjet_clusESampl.push_back(input_tree.GNNjet_clusESampl[jet])
                GNNjet_clusSamplIndex.clear()
                GNNjet_clusSamplIndex.push_back(input_tree.GNNjet_clusSamplIndex[jet])
                GNNjet_clusEta.clear()
                GNNjet_clusEta.push_back(input_tree.GNNjet_clusEta[jet])
                GNNjet_clusPhi.clear()
                GNNjet_clusPhi.push_back(input_tree.GNNjet_clusPhi[jet])
                    
                output_tree.Fill()
            
            if entry%(input_tree.GetEntries()/100) == 0:
                print("entry", str(entry) + '/' + str(input_tree.GetEntries()), "("+str(round(entry/input_tree.GetEntries()*100, 2))+"%)", end='\r')
                
        # Write the output file
        output_root_file.Write()
        output_root_file.Close()
        input_root_file.Close()
        
        print("Flattened file created: ", self.output_file)
        return self.output_file