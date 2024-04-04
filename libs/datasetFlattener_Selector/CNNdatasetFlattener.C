#define CNNdatasetFlattener_cxx

#include "CNNdatasetFlattener.h"
#include <TH2.h>
#include <TStyle.h>

#include <vector>


const TString base_path = "/nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/";

TString output_name;	
TFile *output_file;

TTree *output_tree;

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

void CNNdatasetFlattener::Begin(TTree * /*tree*/)
{
	TString option = GetOption();
}


void CNNdatasetFlattener::SlaveBegin(TTree * /*tree*/)
{
	TString option = GetOption();

	output_name = base_path + "/data/flat/signal.root";


	gInterpreter->GenerateDictionary("vector<vector<float> >", "vector");
	gInterpreter->GenerateDictionary("vector<vector<int> >", "vector");
	gInterpreter->GenerateDictionary("vector<vector<vector<float> >>", "vector");
	gInterpreter->GenerateDictionary("vector<vector<vector<int> >>", "vector");


	output_file = new TFile(output_name, "RECREATE");
	output_tree = new TTree("T", "T");

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
}


Bool_t CNNdatasetFlattener::Process(Long64_t entry)
{
	fReader.SetLocalEntry(entry);

	for (int i = 0; i < GNNjet_width.GetSize(); i++){
		// set all event level variables
		
		isMC_ = *isMC;
		RunNumber_ = *RunNumber;
		dsid_ = *dsid;
		eventNumber_ = *eventNumber;
		amiXsection_ = *amiXsection;
		filterEff_ = *filterEff;
		kFactor_ = *kFactor;
		sumWeightPRW_ = *sumWeightPRW;
		lead_index_ = *lead_index;
		far20_index_ = *far20_index;

		weight_ = *weight;
		puWeight_ = *puWeight;
		mcWeight_ = *mcWeight;
		
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

		for (int i = 0; i < etaLJ.GetSize(); i++){
			etaLJ_.push_back(etaLJ[i]);
		}
		for (int i = 0; i < phiLJ.GetSize(); i++){
			phiLJ_.push_back(phiLJ[i]);
		}
		for (int i = 0; i < ptLJ.GetSize(); i++){
			ptLJ_.push_back(ptLJ[i]);
		}
		for (int i = 0; i < types.GetSize(); i++){
			types_.push_back(types[i]);
		}
		for (int i = 0; i < isoID.GetSize(); i++){
			isoID_.push_back(isoID[i]);
		}
		for (int i = 0; i < LJ_index.GetSize(); i++){
			LJ_index_.push_back(LJ_index[i]);
		}
		for (int i = 0; i < LJ_MatchedTruthDPindex.GetSize(); i++){
			for (int j = 0; j < LJ_MatchedTruthDPindex[i].size(); j++){
				LJ_MatchedTruthDPindex_[i].push_back(LJ_MatchedTruthDPindex[i][j]);
			}
		}
		//for (int i = 0; i < LJjet_index.GetSize(); i++){
		//	LJjet_index_.push_back(LJjet_index[i]);
		//}
		for (int i = 0; i < LJjet_eta.GetSize(); i++){
			LJjet_eta_.push_back(LJjet_eta[i]);
		}
		for (int i = 0; i < LJjet_phi.GetSize(); i++){
			LJjet_phi_.push_back(LJjet_phi[i]);
		}
		for (int i = 0; i < LJjet_pt.GetSize(); i++){
			LJjet_pt_.push_back(LJjet_pt[i]);
		}
		for (int i = 0; i < LJjet_width.GetSize(); i++){
			LJjet_width_.push_back(LJjet_width[i]);
		}
		for (int i = 0; i < LJjet_EMfrac.GetSize(); i++){
			LJjet_EMfrac_.push_back(LJjet_EMfrac[i]);
		}
		for (int i = 0; i < LJjet_timing.GetSize(); i++){
			LJjet_timing_.push_back(LJjet_timing[i]);
		}
		for (int i = 0; i < LJjet_jvt.GetSize(); i++){
			LJjet_jvt_.push_back(LJjet_jvt[i]);
		}
		for (int i = 0; i < LJjet_gapRatio.GetSize(); i++){
			LJjet_gapRatio_.push_back(LJjet_gapRatio[i]);
		}
		//for (int i = 0; i < LJjet_IsBIB.GetSize(); i++){
		//	LJjet_IsBIB_.push_back(LJjet_IsBIB[i]);
		//}
		for (int i = 0; i < LJjet_m.GetSize(); i++){
			LJjet_m_.push_back(LJjet_m[i]);
		}
		for (int i = 0; i < truthPdgId.GetSize(); i++){
			truthPdgId_.push_back(truthPdgId[i]);
		}
		for (int i = 0; i < truthEta.GetSize(); i++){
			truthEta_.push_back(truthEta[i]);
		}
		for (int i = 0; i < truthPhi.GetSize(); i++){
			truthPhi_.push_back(truthPhi[i]);
		}
		for (int i = 0; i < truthPt.GetSize(); i++){
			truthPt_.push_back(truthPt[i]);
		}
		for (int i = 0; i < truthE.GetSize(); i++){
			truthE_.push_back(truthE[i]);
		}
		for (int i = 0; i < truthCharge.GetSize(); i++){
			truthCharge_.push_back(truthCharge[i]);
		}
		for (int i = 0; i < truthBarcode.GetSize(); i++){
			truthBarcode_.push_back(truthBarcode[i]);
		}
		for (int i = 0; i < truthDecayVtx_x.GetSize(); i++){
			truthDecayVtx_x_.push_back(truthDecayVtx_x[i]);
		}
		for (int i = 0; i < truthDecayVtx_y.GetSize(); i++){
			truthDecayVtx_y_.push_back(truthDecayVtx_y[i]);
		}
		for (int i = 0; i < truthDecayVtx_z.GetSize(); i++){
			truthDecayVtx_z_.push_back(truthDecayVtx_z[i]);
		}
		for (int i = 0; i < truthDecayType.GetSize(); i++){
			truthDecayType_.push_back(truthDecayType[i]);
		}
		for (int i = 0; i < childPdgId.GetSize(); i++){
			childPdgId_.push_back(childPdgId[i]);
		}
		for (int i = 0; i < childEta.GetSize(); i++){
			childEta_.push_back(childEta[i]);
		}
		for (int i = 0; i < childPhi.GetSize(); i++){
			childPhi_.push_back(childPhi[i]);
		}
		for (int i = 0; i < childPt.GetSize(); i++){
			childPt_.push_back(childPt[i]);
		}
		for (int i = 0; i < childBarcode.GetSize(); i++){
			childBarcode_.push_back(childBarcode[i]);
		}
		for (int i = 0; i < childMomBarcode.GetSize(); i++){
			childMomBarcode_.push_back(childMomBarcode[i]);
		}
		/*
		for (int i = 0; i < GNNjet_eta.GetSize(); i++){
			GNNjet_eta_.push_back(GNNjet_eta[i]);
		}
		for (int i = 0; i < GNNjet_phi.GetSize(); i++){
			GNNjet_phi_.push_back(GNNjet_phi[i]);
		}
		for (int i = 0; i < GNNjet_pt.GetSize(); i++){
			GNNjet_pt_.push_back(GNNjet_pt[i]);
		}
		for (int i = 0; i < GNNjet_width.GetSize(); i++){
			GNNjet_width_.push_back(GNNjet_width[i]);
		}
		for (int i = 0; i < GNNjet_EMfrac.GetSize(); i++){
			GNNjet_EMfrac_.push_back(GNNjet_EMfrac[i]);
		}
		for (int i = 0; i < GNNjet_timing.GetSize(); i++){
			GNNjet_timing_.push_back(GNNjet_timing[i]);
		}
		for (int i = 0; i < GNNjet_jvt.GetSize(); i++){
			GNNjet_jvt_.push_back(GNNjet_jvt[i]);
		}
		for (int i = 0; i < GNNjet_gapRatio.GetSize(); i++){
			GNNjet_gapRatio_.push_back(GNNjet_gapRatio[i]);
		}
		for (int i = 0; i < GNNjet_m.GetSize(); i++){
			GNNjet_m_.push_back(GNNjet_m[i]);
		}
		for (int i = 0; i < GNNjet_clusESampl.GetSize(); i++){
			GNNjet_clusESampl_.push_back(GNNjet_clusESampl[i]);
		}
		for (int i = 0; i < GNNjet_clusSamplIndex.GetSize(); i++){
			GNNjet_clusSamplIndex_.push_back(GNNjet_clusSamplIndex[i]);
		}
		for (int i = 0; i < GNNjet_clusEta.GetSize(); i++){
			GNNjet_clusEta_.push_back(GNNjet_clusEta[i]);
		}
		for (int i = 0; i < GNNjet_clusPhi.GetSize(); i++){
			GNNjet_clusPhi_.push_back(GNNjet_clusPhi[i]);
		}
		*/


		// set all jet level variables
		for (int i = 0; i < GNNjet_width.GetSize(); i++){
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

			LJjet_index_.push_back(LJjet_index[i]);
			/*LJjet_IsBIB_.push_back(LJjet_IsBIB[i]);
			GNNjet_eta_.push_back(GNNjet_eta[i]);
			GNNjet_phi_.push_back(GNNjet_phi[i]);
			GNNjet_pt_.push_back(GNNjet_pt[i]);
			GNNjet_width_.push_back(GNNjet_width[i]);
			GNNjet_EMfrac_.push_back(GNNjet_EMfrac[i]);
			GNNjet_timing_.push_back(GNNjet_timing[i]);
			GNNjet_jvt_.push_back(GNNjet_jvt[i]);
			GNNjet_gapRatio_.push_back(GNNjet_gapRatio[i]);
			GNNjet_m_.push_back(GNNjet_m[i]);
			GNNjet_clusESampl_.push_back(GNNjet_clusESampl[i]);
			GNNjet_clusSamplIndex_.push_back(GNNjet_clusSamplIndex[i]);
			GNNjet_clusEta_.push_back(GNNjet_clusEta[i]);
			GNNjet_clusPhi_.push_back(GNNjet_clusPhi[i]);*/

			/*LJjet_index_->clear();
			LJjet_IsBIB_->clear();
			GNNjet_eta_->clear();
			GNNjet_phi_->clear();
			GNNjet_pt_->clear();
			GNNjet_width_->clear();
			GNNjet_EMfrac_->clear();
			GNNjet_timing_->clear();
			GNNjet_jvt_->clear();
			GNNjet_gapRatio_->clear();
			GNNjet_m_->clear();
			GNNjet_clusESampl_->clear();
			GNNjet_clusSamplIndex_->clear();
			GNNjet_clusEta_->clear();
			GNNjet_clusPhi_->clear();

			LJjet_index_->push_back(LJjet_index[i]);
			LJjet_IsBIB_->push_back(LJjet_IsBIB[i]);
			GNNjet_eta_->push_back(GNNjet_eta[i]);
			GNNjet_phi_->push_back(GNNjet_phi[i]);
			GNNjet_pt_->push_back(GNNjet_pt[i]);
			GNNjet_width_->push_back(GNNjet_width[i]);
			GNNjet_EMfrac_->push_back(GNNjet_EMfrac[i]);
			GNNjet_timing_->push_back(GNNjet_timing[i]);
			GNNjet_jvt_->push_back(GNNjet_jvt[i]);
			GNNjet_gapRatio_->push_back(GNNjet_gapRatio[i]);
			GNNjet_m_->push_back(GNNjet_m[i]);
			GNNjet_clusESampl_->push_back(GNNjet_clusESampl[i]);
			GNNjet_clusSamplIndex_->push_back(GNNjet_clusSamplIndex[i]);
			GNNjet_clusEta_->push_back(GNNjet_clusEta[i]);
			GNNjet_clusPhi_->push_back(GNNjet_clusPhi[i]);*/
		}
		output_tree->Fill(); //fill once per jet
	}
	return kTRUE;
}


void CNNdatasetFlattener::SlaveTerminate()
{
	output_file->Write();
	output_file->Close();
}

void CNNdatasetFlattener::Terminate()
{

}