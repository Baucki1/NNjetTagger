#define CNNdatasetFlattener_mc_cxx
#include "CNNdatasetFlattener_mc.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <vector>

void CNNdatasetFlattener_mc::Loop()
{
	//   In a ROOT session, you can do:
	//      root> .L CNNdatasetFlattener_mc.C
	//      root> CNNdatasetFlattener_mc t
	//      root> t.GetEntry(12); // Fill t data members with entry number 12
	//      root> t.Show();       // Show values of entry 12
	//      root> t.Show(16);     // Read and show values of entry 16
	//      root> t.Loop();       // Loop on all entries
	//

	//     This is the loop skeleton where:
	//    jentry is the global entry number in the chain
	//    ientry is the entry number in the current Tree
	//  Note that the argument to GetEntry must be:
	//    jentry for TChain::GetEntry
	//    ientry for TTree::GetEntry and TBranch::GetEntry
	//
	//       To read only selected branches, Insert statements like:
	// METHOD1:
	//    fChain->SetBranchStatus("*",0);  // disable all branches
	//    fChain->SetBranchStatus("branchname",1);  // activate branchname
	// METHOD2: replace line
	//    fChain->GetEntry(jentry);       //read all branches
	//by  b_branchname->GetEntry(ientry); //read only this branch
	if (fChain == 0) return;

	Long64_t nentries = fChain->GetEntriesFast();

	const TString base_path = "/nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/";
	TString output_name = base_path + "/data/flat/signal.root";
	
	TFile *output_file = new TFile(output_name, "RECREATE");
	TTree *output_tree = new TTree("T", "T");

	bool isMC_;
	int RunNumber_;
	int dsid_;
	int eventNumber_;
	float amiXsection_;
	float filterEff_;
	float kFactor_;
	float sumWeightPRW_;
	int lead_index_;
	int far20_index_;
	std::vector<float> etaLJ_;
	std::vector<float> phiLJ_;
	std::vector<float> ptLJ_;
	std::vector<int> types_;
	std::vector<float> isoID_;
	std::vector<int> LJ_index_;
	std::vector<std::vector<int>> LJ_MatchedTruthDPindex_;
	float weight_;
	float puWeight_;
	float mcWeight_;
	std::vector<int> LJjet_index_;
	std::vector<float> LJjet_eta_;
	std::vector<float> LJjet_phi_;
	std::vector<float> LJjet_pt_;
	std::vector<float> LJjet_width_;
	std::vector<float> LJjet_EMfrac_;
	std::vector<float> LJjet_timing_;
	std::vector<float> LJjet_jvt_;
	std::vector<float> LJjet_gapRatio_;
	std::vector<float> LJjet_IsBIB_;
	std::vector<float> LJjet_m_;
	std::vector<int> truthPdgId_;
	std::vector<float> truthEta_;
	std::vector<float> truthPhi_;
	std::vector<float> truthPt_;
	std::vector<float> truthE_;
	std::vector<float> truthCharge_;
	std::vector<int> truthBarcode_;
	std::vector<float> truthDecayVtx_x_;
	std::vector<float> truthDecayVtx_y_;
	std::vector<float> truthDecayVtx_z_;
	std::vector<int> truthDecayType_;
	std::vector<int> childPdgId_;
	std::vector<float> childEta_;
	std::vector<float> childPhi_;
	std::vector<float> childPt_;
	std::vector<int> childBarcode_;
	std::vector<int> childMomBarcode_;
	std::vector<float> GNNjet_eta_;
	std::vector<float> GNNjet_phi_;
	std::vector<float> GNNjet_pt_;
	std::vector<float> GNNjet_width_;
	std::vector<float> GNNjet_EMfrac_;
	std::vector<float> GNNjet_timing_;
	std::vector<float> GNNjet_jvt_;
	std::vector<float> GNNjet_gapRatio_;
	std::vector<float> GNNjet_m_;
	std::vector<std::vector<std::vector<float>>> GNNjet_clusESampl_;
	std::vector<std::vector<std::vector<int>>> GNNjet_clusSamplIndex_;
	std::vector<std::vector<float>> GNNjet_clusEta_;
	std::vector<std::vector<float>> GNNjet_clusPhi_;


	gInterpreter->GenerateDictionary("vector<vector<float> >", "vector");
	gInterpreter->GenerateDictionary("vector<vector<int> >", "vector");
	gInterpreter->GenerateDictionary("vector<vector<vector<float> >>", "vector");
	gInterpreter->GenerateDictionary("vector<vector<vector<int> >>", "vector");


	output_tree->Branch("isMC", &isMC_);
	output_tree->Branch("RunNumber", &RunNumber_);
	output_tree->Branch("dsid", &dsid_);
	output_tree->Branch("eventNumber", &eventNumber_);
	output_tree->Branch("amiXsection", &amiXsection_);
	output_tree->Branch("filterEff", &filterEff_);
	output_tree->Branch("kFactor", &kFactor_);
	output_tree->Branch("sumWeightPRW", &sumWeightPRW_);
	output_tree->Branch("lead_index", &lead_index_);
	output_tree->Branch("far20_index", &far20_index_);
	output_tree->Branch("etaLJ", &etaLJ_);
	output_tree->Branch("phiLJ", &phiLJ_);
	output_tree->Branch("ptLJ", &ptLJ_);
	output_tree->Branch("types", &types_);
	output_tree->Branch("isoID", &isoID_);
	output_tree->Branch("LJ_index", &LJ_index_);
	output_tree->Branch("LJ_MatchedTruthDPindex", &LJ_MatchedTruthDPindex_);
	output_tree->Branch("weight", &weight_);
	output_tree->Branch("puWeight", &puWeight_);
	output_tree->Branch("mcWeight", &mcWeight_);
	output_tree->Branch("LJjet_index", &LJjet_index_);
	output_tree->Branch("LJjet_eta", &LJjet_eta_);
	output_tree->Branch("LJjet_phi", &LJjet_phi_);
	output_tree->Branch("LJjet_pt", &LJjet_pt_);
	output_tree->Branch("LJjet_width", &LJjet_width_);
	output_tree->Branch("LJjet_EMfrac", &LJjet_EMfrac_);
	output_tree->Branch("LJjet_timing", &LJjet_timing_);
	output_tree->Branch("LJjet_jvt", &LJjet_jvt_);
	output_tree->Branch("LJjet_gapRatio", &LJjet_gapRatio_);
	output_tree->Branch("LJjet_IsBIB", &LJjet_IsBIB_);
	output_tree->Branch("LJjet_m", &LJjet_m_);
	output_tree->Branch("truthPdgId", &truthPdgId_);
	output_tree->Branch("truthEta", &truthEta_);
	output_tree->Branch("truthPhi", &truthPhi_);
	output_tree->Branch("truthPt", &truthPt_);
	output_tree->Branch("truthE", &truthE_);
	output_tree->Branch("truthCharge", &truthCharge_);
	output_tree->Branch("truthBarcode", &truthBarcode_);
	output_tree->Branch("truthDecayVtx_x", &truthDecayVtx_x_);
	output_tree->Branch("truthDecayVtx_y", &truthDecayVtx_y_);
	output_tree->Branch("truthDecayVtx_z", &truthDecayVtx_z_);
	output_tree->Branch("truthDecayType", &truthDecayType_);
	output_tree->Branch("childPdgId", &childPdgId_);
	output_tree->Branch("childEta", &childEta_);
	output_tree->Branch("childPhi", &childPhi_);
	output_tree->Branch("childPt", &childPt_);
	output_tree->Branch("childBarcode", &childBarcode_);
	output_tree->Branch("childMomBarcode", &childMomBarcode_);
	output_tree->Branch("GNNjet_eta", &GNNjet_eta_);
	output_tree->Branch("GNNjet_phi", &GNNjet_phi_);
	output_tree->Branch("GNNjet_pt", &GNNjet_pt_);
	output_tree->Branch("GNNjet_width", &GNNjet_width_);
	output_tree->Branch("GNNjet_EMfrac", &GNNjet_EMfrac_);
	output_tree->Branch("GNNjet_timing", &GNNjet_timing_);
	output_tree->Branch("GNNjet_jvt", &GNNjet_jvt_);
	output_tree->Branch("GNNjet_gapRatio", &GNNjet_gapRatio_);
	output_tree->Branch("GNNjet_m", &GNNjet_m_);
	output_tree->Branch("GNNjet_clusESampl", &GNNjet_clusESampl_);
	output_tree->Branch("GNNjet_clusSamplIndex", &GNNjet_clusSamplIndex_);
	output_tree->Branch("GNNjet_clusEta", &GNNjet_clusEta_);
	output_tree->Branch("GNNjet_clusPhi", &GNNjet_clusPhi_);

	Long64_t nbytes = 0, nb = 0;
	for (Long64_t jentry=0; jentry<nentries;jentry++) {
		Long64_t ientry = LoadTree(jentry);
		if (ientry < 0) break;
		nb = fChain->GetEntry(jentry);   nbytes += nb;


		// set all event level variables
		isMC_ = isMC;
		RunNumber_ = RunNumber;
		dsid_ = dsid;
		eventNumber_ = eventNumber;
		amiXsection_ = amiXsection;
		filterEff_ = filterEff;
		kFactor_ = kFactor;
		sumWeightPRW_ = sumWeightPRW;
		lead_index_ = lead_index;
		far20_index_ = far20_index;

		weight_ = weight;
		puWeight_ = puWeight;
		mcWeight_ = mcWeight;
		

		for (int i = 0; i < etaLJ->size(); i++){
			etaLJ_.push_back(etaLJ->at(i));
		}
		for (int i = 0; i < phiLJ->size(); i++){
			phiLJ_.push_back(phiLJ->at(i));
		}
		for (int i = 0; i < ptLJ->size(); i++){
			ptLJ_.push_back(ptLJ->at(i));
		}
		for (int i = 0; i < types->size(); i++){
			types_.push_back(types->at(i));
		}
		for (int i = 0; i < isoID->size(); i++){
			isoID_.push_back(isoID->at(i));
		}
		for (int i = 0; i < LJ_index->size(); i++){
			LJ_index_.push_back(LJ_index->at(i));
		}
		for (int i = 0; i < LJ_MatchedTruthDPindex->size(); i++){
			for (int j = 0; j < LJ_MatchedTruthDPindex->at(i).size(); j++){
				LJ_MatchedTruthDPindex_.at(i).push_back(LJ_MatchedTruthDPindex->at(i).at(j));
			}
		}
		//for (int i = 0; i < LJjet_index->size(); i++){
		//	LJjet_index_.push_back(LJjet_index->at(i));
		//}
		for (int i = 0; i < LJjet_eta->size(); i++){
			LJjet_eta_.push_back(LJjet_eta->at(i));
		}
		for (int i = 0; i < LJjet_phi->size(); i++){
			LJjet_phi_.push_back(LJjet_phi->at(i));
		}
		for (int i = 0; i < LJjet_pt->size(); i++){
			LJjet_pt_.push_back(LJjet_pt->at(i));
		}
		for (int i = 0; i < LJjet_width->size(); i++){
			LJjet_width_.push_back(LJjet_width->at(i));
		}
		for (int i = 0; i < LJjet_EMfrac->size(); i++){
			LJjet_EMfrac_.push_back(LJjet_EMfrac->at(i));
		}
		for (int i = 0; i < LJjet_timing->size(); i++){
			LJjet_timing_.push_back(LJjet_timing->at(i));
		}
		for (int i = 0; i < LJjet_jvt->size(); i++){
			LJjet_jvt_.push_back(LJjet_jvt->at(i));
		}
		for (int i = 0; i < LJjet_gapRatio->size(); i++){
			LJjet_gapRatio_.push_back(LJjet_gapRatio->at(i));
		}
		//for (int i = 0; i < LJjet_IsBIB->size(); i++){
		//	LJjet_IsBIB_.push_back(LJjet_IsBIB->at(i));
		//}
		for (int i = 0; i < LJjet_m->size(); i++){
			LJjet_m_.push_back(LJjet_m->at(i));
		}
		for (int i = 0; i < truthPdgId->size(); i++){
			truthPdgId_.push_back(truthPdgId->at(i));
		}
		for (int i = 0; i < truthEta->size(); i++){
			truthEta_.push_back(truthEta->at(i));
		}
		for (int i = 0; i < truthPhi->size(); i++){
			truthPhi_.push_back(truthPhi->at(i));
		}
		for (int i = 0; i < truthPt->size(); i++){
			truthPt_.push_back(truthPt->at(i));
		}
		for (int i = 0; i < truthE->size(); i++){
			truthE_.push_back(truthE->at(i));
		}
		for (int i = 0; i < truthCharge->size(); i++){
			truthCharge_.push_back(truthCharge->at(i));
		}
		for (int i = 0; i < truthBarcode->size(); i++){
			truthBarcode_.push_back(truthBarcode->at(i));
		}
		for (int i = 0; i < truthDecayVtx_x->size(); i++){
			truthDecayVtx_x_.push_back(truthDecayVtx_x->at(i));
		}
		for (int i = 0; i < truthDecayVtx_y->size(); i++){
			truthDecayVtx_y_.push_back(truthDecayVtx_y->at(i));
		}
		for (int i = 0; i < truthDecayVtx_z->size(); i++){
			truthDecayVtx_z_.push_back(truthDecayVtx_z->at(i));
		}
		for (int i = 0; i < truthDecayType->size(); i++){
			truthDecayType_.push_back(truthDecayType->at(i));
		}
		for (int i = 0; i < childPdgId->size(); i++){
			childPdgId_.push_back(childPdgId->at(i));
		}
		for (int i = 0; i < childEta->size(); i++){
			childEta_.push_back(childEta->at(i));
		}
		for (int i = 0; i < childPhi->size(); i++){
			childPhi_.push_back(childPhi->at(i));
		}
		for (int i = 0; i < childPt->size(); i++){
			childPt_.push_back(childPt->at(i));
		}
		for (int i = 0; i < childBarcode->size(); i++){
			childBarcode_.push_back(childBarcode->at(i));
		}
		for (int i = 0; i < childMomBarcode->size(); i++){
			childMomBarcode_.push_back(childMomBarcode->at(i));
		}
		/*
		for (int i = 0; i < GNNjet_eta->size(); i++){
			GNNjet_eta_.push_back(GNNjet_eta->at(i));
		}
		for (int i = 0; i < GNNjet_phi->size(); i++){
			GNNjet_phi_.push_back(GNNjet_phi->at(i));
		}
		for (int i = 0; i < GNNjet_pt->size(); i++){
			GNNjet_pt_.push_back(GNNjet_pt->at(i));
		}
		for (int i = 0; i < GNNjet_width->size(); i++){
			GNNjet_width_.push_back(GNNjet_width->at(i));
		}
		for (int i = 0; i < GNNjet_EMfrac->size(); i++){
			GNNjet_EMfrac_.push_back(GNNjet_EMfrac->at(i));
		}
		for (int i = 0; i < GNNjet_timing->size(); i++){
			GNNjet_timing_.push_back(GNNjet_timing->at(i));
		}
		for (int i = 0; i < GNNjet_jvt->size(); i++){
			GNNjet_jvt_.push_back(GNNjet_jvt->at(i));
		}
		for (int i = 0; i < GNNjet_gapRatio->size(); i++){
			GNNjet_gapRatio_.push_back(GNNjet_gapRatio->at(i));
		}
		for (int i = 0; i < GNNjet_m->size(); i++){
			GNNjet_m_.push_back(GNNjet_m->at(i));
		}
		for (int i = 0; i < GNNjet_clusESampl->size(); i++){
			GNNjet_clusESampl_.push_back(GNNjet_clusESampl->at(i));
		}
		for (int i = 0; i < GNNjet_clusSamplIndex->size(); i++){
			GNNjet_clusSamplIndex_.push_back(GNNjet_clusSamplIndex->at(i));
		}
		for (int i = 0; i < GNNjet_clusEta->size(); i++){
			GNNjet_clusEta_.push_back(GNNjet_clusEta->at(i));
		}
		for (int i = 0; i < GNNjet_clusPhi->size(); i++){
			GNNjet_clusPhi_.push_back(GNNjet_clusPhi->at(i));
		}
		*/


		// set all jet level variables
		for (int i = 0; i < GNNjet_width->size(); i++){
			LJjet_index_.push_back(LJjet_index->at(i));
			LJjet_IsBIB_.push_back(LJjet_IsBIB->at(i));
			GNNjet_eta_.push_back(GNNjet_eta->at(i));
			GNNjet_phi_.push_back(GNNjet_phi->at(i));
			GNNjet_pt_.push_back(GNNjet_pt->at(i));
			GNNjet_width_.push_back(GNNjet_width->at(i));
			GNNjet_EMfrac_.push_back(GNNjet_EMfrac->at(i));
			GNNjet_timing_.push_back(GNNjet_timing->at(i));
			GNNjet_jvt_.push_back(GNNjet_jvt->at(i));
			GNNjet_gapRatio_.push_back(GNNjet_gapRatio->at(i));
			GNNjet_m_.push_back(GNNjet_m->at(i));
			for (int j = 0; j < GNNjet_clusESampl->at(i).size(); j++){
				for (int k = 0; k < GNNjet_clusESampl->at(i).at(j).size(); k++){
					GNNjet_clusESampl_.at(i).at(j).push_back(GNNjet_clusESampl->at(i).at(j).at(k));
				}
			}
			for (int j = 0; j < GNNjet_clusSamplIndex->at(i).size(); j++){
				for (int k = 0; k < GNNjet_clusSamplIndex->at(i).at(j).size(); k++){
					GNNjet_clusSamplIndex_.at(i).at(j).push_back(GNNjet_clusSamplIndex->at(i).at(j).at(k));
				}
			}
			for (int j = 0; j < GNNjet_clusEta->at(i).size(); j++){
				GNNjet_clusEta_.at(i).push_back(GNNjet_clusEta->at(i).at(j));
			}
			for (int j = 0; j < GNNjet_clusPhi->at(i).size(); j++){
				GNNjet_clusPhi_.at(i).push_back(GNNjet_clusPhi->at(i).at(j));
			}
			
			output_tree->Fill(); //fill once per jet

			LJjet_index_.clear();
			LJjet_IsBIB_.clear();
			GNNjet_eta_.clear();
			GNNjet_phi_.clear();
			GNNjet_pt_.clear();
			GNNjet_width_.clear();
			GNNjet_EMfrac_.clear();
			GNNjet_timing_.clear();
			GNNjet_jvt_.clear();
			GNNjet_gapRatio_.clear();
			GNNjet_m_.clear();
			GNNjet_clusESampl_.clear();
			GNNjet_clusSamplIndex_.clear();
			GNNjet_clusEta_.clear();
			GNNjet_clusPhi_.clear();
		}


		etaLJ_.clear();
		phiLJ_.clear();
		ptLJ_.clear();
		types_.clear();
		isoID_.clear();
		LJ_index_.clear();
		//LJjet_index_.clear();
		LJjet_eta_.clear();
		LJjet_phi_.clear();
		LJjet_pt_.clear();
		LJjet_width_.clear();
		LJjet_EMfrac_.clear();
		LJjet_timing_.clear();
		LJjet_jvt_.clear();
		LJjet_gapRatio_.clear();
		//LJjet_IsBIB_.clear();
		LJjet_m_.clear();
		truthPdgId_.clear();
		truthEta_.clear();
		truthPhi_.clear();
		truthPt_.clear();
		truthE_.clear();
		truthCharge_.clear();
		truthBarcode_.clear();
		truthDecayVtx_x_.clear();
		truthDecayVtx_y_.clear();
		truthDecayVtx_z_.clear();
		truthDecayType_.clear();
		childPdgId_.clear();
		childEta_.clear();
		childPhi_.clear();
		childPt_.clear();
		childBarcode_.clear();
		childMomBarcode_.clear();
		//GNNjet_eta_.clear();
		//GNNjet_phi_.clear();
		//GNNjet_pt_.clear();
		//GNNjet_width_.clear();
		//GNNjet_EMfrac_.clear();
		//GNNjet_timing_.clear();
		//GNNjet_jvt_.clear();
		//GNNjet_gapRatio_.clear();
		//GNNjet_m_.clear();
		//GNNjet_clusESampl_.clear();
		//GNNjet_clusSamplIndex_.clear();
		//GNNjet_clusEta_.clear();
		//GNNjet_clusPhi_.clear();
	}
}
