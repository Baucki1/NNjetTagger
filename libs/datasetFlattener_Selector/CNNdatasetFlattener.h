//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Mar 19 16:05:37 2024 by ROOT version 6.24/08
// from TTree T/skimmed T
// found on file: ../../data/signal.root
//////////////////////////////////////////////////////////

#ifndef CNNdatasetFlattener_h
#define CNNdatasetFlattener_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

// Headers needed by this particular selector
#include <vector>


class CNNdatasetFlattener : public TSelector {
public :
   TTreeReader     fReader;  //!the tree reader
   TTree          *fChain = 0;   //!pointer to the analyzed TTree or TChain

   // Readers to access the data (delete the ones you do not need).
   TTreeReaderValue<Bool_t> isMC = {fReader, "isMC"};
   TTreeReaderValue<Int_t> RunNumber = {fReader, "RunNumber"};
   TTreeReaderValue<Int_t> dsid = {fReader, "dsid"};
   TTreeReaderValue<Int_t> eventNumber = {fReader, "eventNumber"};
   TTreeReaderValue<Float_t> amiXsection = {fReader, "amiXsection"};
   TTreeReaderValue<Float_t> filterEff = {fReader, "filterEff"};
   TTreeReaderValue<Float_t> kFactor = {fReader, "kFactor"};
   TTreeReaderValue<Float_t> sumWeightPRW = {fReader, "sumWeightPRW"};
   TTreeReaderValue<Int_t> lead_index = {fReader, "lead_index"};
   TTreeReaderValue<Int_t> far20_index = {fReader, "far20_index"};
   TTreeReaderArray<float> etaLJ = {fReader, "etaLJ"};
   TTreeReaderArray<float> phiLJ = {fReader, "phiLJ"};
   TTreeReaderArray<float> ptLJ = {fReader, "ptLJ"};
   TTreeReaderArray<int> types = {fReader, "types"};
   TTreeReaderArray<float> isoID = {fReader, "isoID"};
   TTreeReaderArray<int> LJ_index = {fReader, "LJ_index"};
   TTreeReaderArray<vector<int>> LJ_MatchedTruthDPindex = {fReader, "LJ_MatchedTruthDPindex"};
   TTreeReaderValue<Float_t> weight = {fReader, "weight"};
   TTreeReaderValue<Float_t> puWeight = {fReader, "puWeight"};
   TTreeReaderValue<Float_t> mcWeight = {fReader, "mcWeight"};
   TTreeReaderArray<int> LJjet_index = {fReader, "LJjet_index"};
   TTreeReaderArray<float> LJjet_eta = {fReader, "LJjet_eta"};
   TTreeReaderArray<float> LJjet_phi = {fReader, "LJjet_phi"};
   TTreeReaderArray<float> LJjet_pt = {fReader, "LJjet_pt"};
   TTreeReaderArray<float> LJjet_width = {fReader, "LJjet_width"};
   TTreeReaderArray<float> LJjet_EMfrac = {fReader, "LJjet_EMfrac"};
   TTreeReaderArray<float> LJjet_timing = {fReader, "LJjet_timing"};
   TTreeReaderArray<float> LJjet_jvt = {fReader, "LJjet_jvt"};
   TTreeReaderArray<float> LJjet_gapRatio = {fReader, "LJjet_gapRatio"};
   TTreeReaderArray<float> LJjet_IsBIB = {fReader, "LJjet_IsBIB"};
   TTreeReaderArray<float> LJjet_m = {fReader, "LJjet_m"};
   TTreeReaderArray<int> truthPdgId = {fReader, "truthPdgId"};
   TTreeReaderArray<float> truthEta = {fReader, "truthEta"};
   TTreeReaderArray<float> truthPhi = {fReader, "truthPhi"};
   TTreeReaderArray<float> truthPt = {fReader, "truthPt"};
   TTreeReaderArray<float> truthE = {fReader, "truthE"};
   TTreeReaderArray<float> truthCharge = {fReader, "truthCharge"};
   TTreeReaderArray<int> truthBarcode = {fReader, "truthBarcode"};
   TTreeReaderArray<float> truthDecayVtx_x = {fReader, "truthDecayVtx_x"};
   TTreeReaderArray<float> truthDecayVtx_y = {fReader, "truthDecayVtx_y"};
   TTreeReaderArray<float> truthDecayVtx_z = {fReader, "truthDecayVtx_z"};
   TTreeReaderArray<int> truthDecayType = {fReader, "truthDecayType"};
   TTreeReaderArray<int> childPdgId = {fReader, "childPdgId"};
   TTreeReaderArray<float> childEta = {fReader, "childEta"};
   TTreeReaderArray<float> childPhi = {fReader, "childPhi"};
   TTreeReaderArray<float> childPt = {fReader, "childPt"};
   TTreeReaderArray<int> childBarcode = {fReader, "childBarcode"};
   TTreeReaderArray<int> childMomBarcode = {fReader, "childMomBarcode"};
   TTreeReaderArray<float> GNNjet_eta = {fReader, "GNNjet_eta"};
   TTreeReaderArray<float> GNNjet_phi = {fReader, "GNNjet_phi"};
   TTreeReaderArray<float> GNNjet_pt = {fReader, "GNNjet_pt"};
   TTreeReaderArray<float> GNNjet_width = {fReader, "GNNjet_width"};
   TTreeReaderArray<float> GNNjet_EMfrac = {fReader, "GNNjet_EMfrac"};
   TTreeReaderArray<float> GNNjet_timing = {fReader, "GNNjet_timing"};
   TTreeReaderArray<float> GNNjet_jvt = {fReader, "GNNjet_jvt"};
   TTreeReaderArray<float> GNNjet_gapRatio = {fReader, "GNNjet_gapRatio"};
   TTreeReaderArray<float> GNNjet_m = {fReader, "GNNjet_m"};
   TTreeReaderArray<vector<vector<float> >> GNNjet_clusESampl = {fReader, "GNNjet_clusESampl"};
   TTreeReaderArray<vector<vector<int> >> GNNjet_clusSamplIndex = {fReader, "GNNjet_clusSamplIndex"};
   TTreeReaderArray<vector<float>> GNNjet_clusEta = {fReader, "GNNjet_clusEta"};
   TTreeReaderArray<vector<float>> GNNjet_clusPhi = {fReader, "GNNjet_clusPhi"};


   CNNdatasetFlattener(TTree * /*tree*/ =0) { }
   virtual ~CNNdatasetFlattener() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(CNNdatasetFlattener,0);

};

#endif

#ifdef CNNdatasetFlattener_cxx
void CNNdatasetFlattener::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the reader is initialized.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   fReader.SetTree(tree);
}

Bool_t CNNdatasetFlattener::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}


#endif // #ifdef CNNdatasetFlattener_cxx
