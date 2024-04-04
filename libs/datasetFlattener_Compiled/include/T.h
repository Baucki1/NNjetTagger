//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed Mar 20 14:45:06 2024 by ROOT version 6.24/08
// from TTree T/skimmed T
// found on file: /nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/data/signal.root
//////////////////////////////////////////////////////////

#ifndef T_h
#define T_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

using namespace std;

// Header file for the classes stored in the TTree if any.
#include "vector"

class T {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Bool_t          isMC;
   Int_t           RunNumber;
   Int_t           dsid;
   Int_t           eventNumber;
   Float_t         amiXsection;
   Float_t         filterEff;
   Float_t         kFactor;
   Float_t         sumWeightPRW;
   Int_t           lead_index;
   Int_t           far20_index;
   vector<float>   *etaLJ;
   vector<float>   *phiLJ;
   vector<float>   *ptLJ;
   vector<int>     *types;
   vector<float>   *isoID;
   vector<int>     *LJ_index;
   vector<vector<int> > *LJ_MatchedTruthDPindex;
   Float_t         weight;
   Float_t         puWeight;
   Float_t         mcWeight;
   vector<int>     *LJjet_index;
   vector<float>   *LJjet_eta;
   vector<float>   *LJjet_phi;
   vector<float>   *LJjet_pt;
   vector<float>   *LJjet_width;
   vector<float>   *LJjet_EMfrac;
   vector<float>   *LJjet_timing;
   vector<float>   *LJjet_jvt;
   vector<float>   *LJjet_gapRatio;
   vector<float>   *LJjet_IsBIB;
   vector<float>   *LJjet_m;
   vector<int>     *truthPdgId;
   vector<float>   *truthEta;
   vector<float>   *truthPhi;
   vector<float>   *truthPt;
   vector<float>   *truthE;
   vector<float>   *truthCharge;
   vector<int>     *truthBarcode;
   vector<float>   *truthDecayVtx_x;
   vector<float>   *truthDecayVtx_y;
   vector<float>   *truthDecayVtx_z;
   vector<int>     *truthDecayType;
   vector<int>     *childPdgId;
   vector<float>   *childEta;
   vector<float>   *childPhi;
   vector<float>   *childPt;
   vector<int>     *childBarcode;
   vector<int>     *childMomBarcode;
   vector<float>   *GNNjet_eta;
   vector<float>   *GNNjet_phi;
   vector<float>   *GNNjet_pt;
   vector<float>   *GNNjet_width;
   vector<float>   *GNNjet_EMfrac;
   vector<float>   *GNNjet_timing;
   vector<float>   *GNNjet_jvt;
   vector<float>   *GNNjet_gapRatio;
   vector<float>   *GNNjet_m;
   vector<vector<vector<float> > > *GNNjet_clusESampl;
   vector<vector<vector<int> > > *GNNjet_clusSamplIndex;
   vector<vector<float> > *GNNjet_clusEta;
   vector<vector<float> > *GNNjet_clusPhi;

   // List of branches
   TBranch        *b_isMC;   //!
   TBranch        *b_RunNumber;   //!
   TBranch        *b_dsid;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_amiXsection;   //!
   TBranch        *b_filterEff;   //!
   TBranch        *b_kFactor;   //!
   TBranch        *b_sumWeightPRW;   //!
   TBranch        *b_lead_index;   //!
   TBranch        *b_far20_index;   //!
   TBranch        *b_etaLJ;   //!
   TBranch        *b_phiLJ;   //!
   TBranch        *b_ptLJ;   //!
   TBranch        *b_types;   //!
   TBranch        *b_isoID;   //!
   TBranch        *b_LJ_index;   //!
   TBranch        *b_LJ_MatchedTruthDPindex;   //!
   TBranch        *b_weight;   //!
   TBranch        *b_puWeight;   //!
   TBranch        *b_mcWeight;   //!
   TBranch        *b_LJjet_index;   //!
   TBranch        *b_LJjet_eta;   //!
   TBranch        *b_LJjet_phi;   //!
   TBranch        *b_LJjet_pt;   //!
   TBranch        *b_LJjet_width;   //!
   TBranch        *b_LJjet_EMfrac;   //!
   TBranch        *b_LJjet_timing;   //!
   TBranch        *b_LJjet_jvt;   //!
   TBranch        *b_LJjet_gapRatio;   //!
   TBranch        *b_LJjet_IsBIB;   //!
   TBranch        *b_LJjet_m;   //!
   TBranch        *b_truthPdgId;   //!
   TBranch        *b_truthEta;   //!
   TBranch        *b_truthPhi;   //!
   TBranch        *b_truthPt;   //!
   TBranch        *b_truthE;   //!
   TBranch        *b_truthCharge;   //!
   TBranch        *b_truthBarcode;   //!
   TBranch        *b_truthDecayVtx_x;   //!
   TBranch        *b_truthDecayVtx_y;   //!
   TBranch        *b_truthDecayVtx_z;   //!
   TBranch        *b_truthDecayType;   //!
   TBranch        *b_childPdgId;   //!
   TBranch        *b_childEta;   //!
   TBranch        *b_childPhi;   //!
   TBranch        *b_childPt;   //!
   TBranch        *b_childBarcode;   //!
   TBranch        *b_childMomBarcode;   //!
   TBranch        *b_GNNjet_eta;   //!
   TBranch        *b_GNNjet_phi;   //!
   TBranch        *b_GNNjet_pt;   //!
   TBranch        *b_GNNjet_width;   //!
   TBranch        *b_GNNjet_EMfrac;   //!
   TBranch        *b_GNNjet_timing;   //!
   TBranch        *b_GNNjet_jvt;   //!
   TBranch        *b_GNNjet_gapRatio;   //!
   TBranch        *b_GNNjet_m;   //!
   TBranch        *b_GNNjet_clusESampl;   //!
   TBranch        *b_GNNjet_clusSamplIndex;   //!
   TBranch        *b_GNNjet_clusEta;   //!
   TBranch        *b_GNNjet_clusPhi;   //!

   T(TTree *tree=0);
   virtual ~T();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(std::string outFileName);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef T_cxx
T::T(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/data/signal.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/data/signal.root");
      }
      f->GetObject("T",tree);

   }
   Init(tree);
}

T::~T()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t T::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t T::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void T::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   etaLJ = 0;
   phiLJ = 0;
   ptLJ = 0;
   types = 0;
   isoID = 0;
   LJ_index = 0;
   LJ_MatchedTruthDPindex = 0;
   LJjet_index = 0;
   LJjet_eta = 0;
   LJjet_phi = 0;
   LJjet_pt = 0;
   LJjet_width = 0;
   LJjet_EMfrac = 0;
   LJjet_timing = 0;
   LJjet_jvt = 0;
   LJjet_gapRatio = 0;
   LJjet_IsBIB = 0;
   LJjet_m = 0;
   truthPdgId = 0;
   truthEta = 0;
   truthPhi = 0;
   truthPt = 0;
   truthE = 0;
   truthCharge = 0;
   truthBarcode = 0;
   truthDecayVtx_x = 0;
   truthDecayVtx_y = 0;
   truthDecayVtx_z = 0;
   truthDecayType = 0;
   childPdgId = 0;
   childEta = 0;
   childPhi = 0;
   childPt = 0;
   childBarcode = 0;
   childMomBarcode = 0;
   GNNjet_eta = 0;
   GNNjet_phi = 0;
   GNNjet_pt = 0;
   GNNjet_width = 0;
   GNNjet_EMfrac = 0;
   GNNjet_timing = 0;
   GNNjet_jvt = 0;
   GNNjet_gapRatio = 0;
   GNNjet_m = 0;
   GNNjet_clusESampl = 0;
   GNNjet_clusSamplIndex = 0;
   GNNjet_clusEta = 0;
   GNNjet_clusPhi = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("isMC", &isMC, &b_isMC);
   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);
   fChain->SetBranchAddress("dsid", &dsid, &b_dsid);
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("amiXsection", &amiXsection, &b_amiXsection);
   fChain->SetBranchAddress("filterEff", &filterEff, &b_filterEff);
   fChain->SetBranchAddress("kFactor", &kFactor, &b_kFactor);
   fChain->SetBranchAddress("sumWeightPRW", &sumWeightPRW, &b_sumWeightPRW);
   fChain->SetBranchAddress("lead_index", &lead_index, &b_lead_index);
   fChain->SetBranchAddress("far20_index", &far20_index, &b_far20_index);
   fChain->SetBranchAddress("etaLJ", &etaLJ, &b_etaLJ);
   fChain->SetBranchAddress("phiLJ", &phiLJ, &b_phiLJ);
   fChain->SetBranchAddress("ptLJ", &ptLJ, &b_ptLJ);
   fChain->SetBranchAddress("types", &types, &b_types);
   fChain->SetBranchAddress("isoID", &isoID, &b_isoID);
   fChain->SetBranchAddress("LJ_index", &LJ_index, &b_LJ_index);
   fChain->SetBranchAddress("LJ_MatchedTruthDPindex", &LJ_MatchedTruthDPindex, &b_LJ_MatchedTruthDPindex);
   fChain->SetBranchAddress("weight", &weight, &b_weight);
   fChain->SetBranchAddress("puWeight", &puWeight, &b_puWeight);
   fChain->SetBranchAddress("mcWeight", &mcWeight, &b_mcWeight);
   fChain->SetBranchAddress("LJjet_index", &LJjet_index, &b_LJjet_index);
   fChain->SetBranchAddress("LJjet_eta", &LJjet_eta, &b_LJjet_eta);
   fChain->SetBranchAddress("LJjet_phi", &LJjet_phi, &b_LJjet_phi);
   fChain->SetBranchAddress("LJjet_pt", &LJjet_pt, &b_LJjet_pt);
   fChain->SetBranchAddress("LJjet_width", &LJjet_width, &b_LJjet_width);
   fChain->SetBranchAddress("LJjet_EMfrac", &LJjet_EMfrac, &b_LJjet_EMfrac);
   fChain->SetBranchAddress("LJjet_timing", &LJjet_timing, &b_LJjet_timing);
   fChain->SetBranchAddress("LJjet_jvt", &LJjet_jvt, &b_LJjet_jvt);
   fChain->SetBranchAddress("LJjet_gapRatio", &LJjet_gapRatio, &b_LJjet_gapRatio);
   fChain->SetBranchAddress("LJjet_IsBIB", &LJjet_IsBIB, &b_LJjet_IsBIB);
   fChain->SetBranchAddress("LJjet_m", &LJjet_m, &b_LJjet_m);
   fChain->SetBranchAddress("truthPdgId", &truthPdgId, &b_truthPdgId);
   fChain->SetBranchAddress("truthEta", &truthEta, &b_truthEta);
   fChain->SetBranchAddress("truthPhi", &truthPhi, &b_truthPhi);
   fChain->SetBranchAddress("truthPt", &truthPt, &b_truthPt);
   fChain->SetBranchAddress("truthE", &truthE, &b_truthE);
   fChain->SetBranchAddress("truthCharge", &truthCharge, &b_truthCharge);
   fChain->SetBranchAddress("truthBarcode", &truthBarcode, &b_truthBarcode);
   fChain->SetBranchAddress("truthDecayVtx_x", &truthDecayVtx_x, &b_truthDecayVtx_x);
   fChain->SetBranchAddress("truthDecayVtx_y", &truthDecayVtx_y, &b_truthDecayVtx_y);
   fChain->SetBranchAddress("truthDecayVtx_z", &truthDecayVtx_z, &b_truthDecayVtx_z);
   fChain->SetBranchAddress("truthDecayType", &truthDecayType, &b_truthDecayType);
   fChain->SetBranchAddress("childPdgId", &childPdgId, &b_childPdgId);
   fChain->SetBranchAddress("childEta", &childEta, &b_childEta);
   fChain->SetBranchAddress("childPhi", &childPhi, &b_childPhi);
   fChain->SetBranchAddress("childPt", &childPt, &b_childPt);
   fChain->SetBranchAddress("childBarcode", &childBarcode, &b_childBarcode);
   fChain->SetBranchAddress("childMomBarcode", &childMomBarcode, &b_childMomBarcode);
   fChain->SetBranchAddress("GNNjet_eta", &GNNjet_eta, &b_GNNjet_eta);
   fChain->SetBranchAddress("GNNjet_phi", &GNNjet_phi, &b_GNNjet_phi);
   fChain->SetBranchAddress("GNNjet_pt", &GNNjet_pt, &b_GNNjet_pt);
   fChain->SetBranchAddress("GNNjet_width", &GNNjet_width, &b_GNNjet_width);
   fChain->SetBranchAddress("GNNjet_EMfrac", &GNNjet_EMfrac, &b_GNNjet_EMfrac);
   fChain->SetBranchAddress("GNNjet_timing", &GNNjet_timing, &b_GNNjet_timing);
   fChain->SetBranchAddress("GNNjet_jvt", &GNNjet_jvt, &b_GNNjet_jvt);
   fChain->SetBranchAddress("GNNjet_gapRatio", &GNNjet_gapRatio, &b_GNNjet_gapRatio);
   fChain->SetBranchAddress("GNNjet_m", &GNNjet_m, &b_GNNjet_m);
   fChain->SetBranchAddress("GNNjet_clusESampl", &GNNjet_clusESampl, &b_GNNjet_clusESampl);
   fChain->SetBranchAddress("GNNjet_clusSamplIndex", &GNNjet_clusSamplIndex, &b_GNNjet_clusSamplIndex);
   fChain->SetBranchAddress("GNNjet_clusEta", &GNNjet_clusEta, &b_GNNjet_clusEta);
   fChain->SetBranchAddress("GNNjet_clusPhi", &GNNjet_clusPhi, &b_GNNjet_clusPhi);
   Notify();
}

Bool_t T::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void T::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t T::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef T_cxx
