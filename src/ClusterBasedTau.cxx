// gSystem->SetIncludePath()
#include <iostream>
#include "TLorentzVector.h"
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include "TStopwatch.h"
#include <math.h>

#include "TauDiscriminant/Pi0Finder.h" 
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<vector<float> >+;
#endif


void ClusterBasedTau(TString in_file,bool doCBCalculation=true){
  TFile *fin = new TFile(in_file,"READ");
  TTree *tree = (TTree*)fin->Get("tau");

  int tau_n;
  vector<float> * tau_pt;
  vector<float> * tau_eta;
  vector<float> * tau_phi;
  vector<float> * tau_m;
  vector<int> * tau_numTrack;
  vector<int> * tau_pi0_n;

  vector<float> * tau_pi0_vistau_pt;
  vector<float> * tau_pi0_vistau_eta;
  vector<float> * tau_pi0_vistau_phi;
  vector<float> * tau_pi0_vistau_m;
  
  vector<int>* tau_track_atTJVA_n;
  vector< vector<float> >* tau_track_atTJVA_pt;
  vector< vector<float> >* tau_track_atTJVA_eta;
  vector< vector<float> >* tau_track_atTJVA_phi;

  vector<int>*             tau_cluster_n;
  vector< vector<float> >* tau_cluster_E;
  vector< vector<float> >* tau_cluster_eta_atTJVA;
  vector< vector<float> >* tau_cluster_phi_atTJVA;
  vector<vector<float> > *tau_cluster_PreSamplerStripF;
  vector<vector<float> > *tau_cluster_EMLayer2F;
  vector<vector<float> > *tau_cluster_EMLayer3F;
  
  TBranch * b_tau_n;
  TBranch * b_tau_pt;
  TBranch * b_tau_eta;
  TBranch * b_tau_phi;
  TBranch * b_tau_m;
  TBranch * b_tau_numTrack;
  TBranch * b_tau_pi0_n;
  
  TBranch * b_tau_pi0_vistau_pt;
  TBranch * b_tau_pi0_vistau_eta;
  TBranch * b_tau_pi0_vistau_phi;
  TBranch * b_tau_pi0_vistau_m;
  
  TBranch * b_tau_track_atTJVA_n;
  TBranch * b_tau_track_atTJVA_pt;
  TBranch * b_tau_track_atTJVA_eta;
  TBranch * b_tau_track_atTJVA_phi;

  TBranch        *b_tau_cluster_E;   //!
  TBranch        *b_tau_cluster_eta_atTJVA;   //!
  TBranch        *b_tau_cluster_phi_atTJVA;   //!
  TBranch        *b_tau_cluster_PreSamplerStripF;   //!
  TBranch        *b_tau_cluster_EMLayer2F;   //!
  TBranch        *b_tau_cluster_EMLayer3F;   //!
  TBranch        *b_tau_cluster_n;   //!  

  tau_pt  = 0;
  tau_eta = 0;
  tau_phi = 0;
  tau_m   = 0;
  tau_numTrack = 0;
  tau_pi0_n = 0;

  tau_pi0_vistau_pt = 0;
  tau_pi0_vistau_eta = 0;
  tau_pi0_vistau_phi = 0;
  tau_pi0_vistau_m = 0;
  
  tau_track_atTJVA_n = 0;
  tau_track_atTJVA_pt = 0;
  tau_track_atTJVA_eta = 0;
  tau_track_atTJVA_phi = 0;

  tau_cluster_E = 0;
  tau_cluster_eta_atTJVA = 0;
  tau_cluster_phi_atTJVA = 0;
  tau_cluster_PreSamplerStripF = 0;
  tau_cluster_EMLayer2F = 0;
  tau_cluster_EMLayer3F = 0;
  tau_cluster_n = 0;

  tree->SetBranchAddress( "tau_n", &tau_n, &b_tau_n);
  tree->SetBranchAddress( "tau_pt", &tau_pt, &b_tau_pt);
  tree->SetBranchAddress( "tau_eta", &tau_eta, &b_tau_eta);
  tree->SetBranchAddress( "tau_phi", &tau_phi, &b_tau_phi);
  tree->SetBranchAddress( "tau_m", &tau_m, &b_tau_m);
  tree->SetBranchAddress( "tau_numTrack", &tau_numTrack, &b_tau_numTrack);
  tree->SetBranchAddress( "tau_pi0_n", &tau_pi0_n, &b_tau_pi0_n);
  
//   tree->SetBranchAddress( "tau_pi0_vistau_pt" , &tau_pi0_vistau_pt , &b_tau_pi0_vistau_pt);
//   tree->SetBranchAddress( "tau_pi0_vistau_eta", &tau_pi0_vistau_eta, &b_tau_pi0_vistau_eta);
//   tree->SetBranchAddress( "tau_pi0_vistau_phi", &tau_pi0_vistau_phi, &b_tau_pi0_vistau_phi);
//   tree->SetBranchAddress( "tau_pi0_vistau_m"  , &tau_pi0_vistau_m  , &b_tau_pi0_vistau_m);
  
  tree->SetBranchAddress( "tau_track_atTJVA_n"  , &tau_track_atTJVA_n  , &b_tau_track_atTJVA_n);
  tree->SetBranchAddress( "tau_track_atTJVA_pt" , &tau_track_atTJVA_pt , &b_tau_track_atTJVA_pt);
  tree->SetBranchAddress( "tau_track_atTJVA_eta", &tau_track_atTJVA_eta, &b_tau_track_atTJVA_eta);
  tree->SetBranchAddress( "tau_track_atTJVA_phi", &tau_track_atTJVA_phi, &b_tau_track_atTJVA_phi);


  tree->SetBranchAddress("tau_cluster_E", &tau_cluster_E, &b_tau_cluster_E);
  tree->SetBranchAddress("tau_cluster_eta_atTJVA", &tau_cluster_eta_atTJVA, &b_tau_cluster_eta_atTJVA);
  tree->SetBranchAddress("tau_cluster_phi_atTJVA", &tau_cluster_phi_atTJVA, &b_tau_cluster_phi_atTJVA);
  tree->SetBranchAddress("tau_cluster_PreSamplerStripF", &tau_cluster_PreSamplerStripF, &b_tau_cluster_PreSamplerStripF);
  tree->SetBranchAddress("tau_cluster_EMLayer2F", &tau_cluster_EMLayer2F, &b_tau_cluster_EMLayer2F);
  tree->SetBranchAddress("tau_cluster_EMLayer3F", &tau_cluster_EMLayer3F, &b_tau_cluster_EMLayer3F);
  tree->SetBranchAddress("tau_cluster_n", &tau_cluster_n, &b_tau_cluster_n);

  TStopwatch TSW;
  TSW.Start();
  int nentries = tree->GetEntries();
  int Ntaus = 0;
  for( int entry=0; entry<nentries; entry++){
    tree->GetEntry(entry);
    int ntaus = (int)tau_n;
    Ntaus += ntaus;

    if (!doCBCalculation) continue;
    for ( int itau=0; itau<ntaus; itau++) {


      // ---> Cluster based 4Vec computation
      int numTrack = tau_numTrack->at(itau);
      int nPi0s    = tau_pi0_n->at(itau);
      
      vector<TLorentzVector> clusters;
      vector<float> PSSF;
      vector<float> EM2F;
      vector<float> EM3F;
      vector<TLorentzVector> tracks;
      for( int itrk=0; itrk<tau_track_atTJVA_n->at(itau); itrk++){
	TLorentzVector curTrk;
	curTrk.SetPtEtaPhiM( (tau_track_atTJVA_pt->at(itau))[itrk],
			     (tau_track_atTJVA_eta->at(itau))[itrk],
			     (tau_track_atTJVA_phi->at(itau))[itrk],
			     0 );
	tracks.push_back(curTrk);
      }

      for( int iclus=0; iclus<tau_cluster_n->at(itau); iclus++){
	TLorentzVector curClus;
	curClus.SetPtEtaPhiM( (tau_cluster_E->at(itau))[iclus]/cosh( (tau_cluster_eta_atTJVA->at(itau))[iclus] ),
			      (tau_cluster_eta_atTJVA->at(itau))[iclus],
			      (tau_cluster_phi_atTJVA->at(itau))[iclus],
			      0 );
	clusters.push_back(curClus);
	double curPSSF = (tau_cluster_PreSamplerStripF->at(itau))[iclus];
	double curEM2F = (tau_cluster_EMLayer2F->at(itau))[iclus];
	double curEM3F = (tau_cluster_EMLayer3F->at(itau))[iclus];
	if (curPSSF < 0.) curPSSF = 0.;
	if (curPSSF > 1.) curPSSF = 1.;
	if (curEM2F < 0.) curEM2F = 0.;
	if (curEM2F > 1.) curEM2F = 1.;
	if (curEM3F < 0.) curEM3F = 0.;
	if (curEM3F > 1.) curEM3F = 1.;
	
	PSSF.push_back( curPSSF );
	EM2F.push_back( curEM2F );
	EM3F.push_back( curEM3F );
      }	



      TLorentzVector clbased_4vec;
      if(nPi0s==0){
	if (numTrack>0){
	  TLorentzVector sumTrk;
	  for( int itrk=0;itrk<tau_track_atTJVA_n->at(itau);itrk++){
	    TLorentzVector curTrk;
	    curTrk.SetPtEtaPhiM( (tau_track_atTJVA_pt->at(itau))[itrk],
				 (tau_track_atTJVA_eta->at(itau))[itrk],
				 (tau_track_atTJVA_phi->at(itau))[itrk],
				 139.98 );
	    sumTrk += curTrk;
	  }
	  clbased_4vec.SetPtEtaPhiM( sumTrk.Pt(),sumTrk.Eta(),sumTrk.Phi(),sumTrk.M() );
	} else
	  clbased_4vec.SetPtEtaPhiM ( tau_pt->at(itau), tau_eta->at(itau), tau_phi->at(itau), tau_m->at(itau) );
      } else if(nPi0s==1 || nPi0s==2){
	if (numTrack>0){
	  //--> Pi0 Finder call
	  Pi0Finder pi0F( tracks, clusters, PSSF, EM2F, EM3F);
	  if (pi0F.visTauTLV().Pt()>0) clbased_4vec = pi0F.visTauTLV();
	  else clbased_4vec.SetPtEtaPhiM ( tau_pt->at(itau), tau_eta->at(itau), tau_phi->at(itau), tau_m->at(itau) );
	}
        else{
	  clbased_4vec.SetPtEtaPhiM ( tau_pt->at(itau), tau_eta->at(itau), tau_phi->at(itau), tau_m->at(itau) );
	}
      }
      //--> End of the cluster-based pt calculation

    }//--> End of the loop over the taus

  }//--> End of the loop over the entries
  TSW.Stop();
  TSW.Print();
  cout << "Nevents = " << nentries << endl;
  cout << "Ntaus = " << Ntaus << endl;
  cout << "Real time (s) per event = " << TSW.RealTime()/(double)nentries << endl;
  cout << "CPU  time (s) per event = " << TSW.CpuTime()/(double)nentries  << endl;
  cout << "Real time (s) per tau = " << TSW.RealTime()/(double)Ntaus << endl;
  cout << "CPU  time (s) per tau = " << TSW.CpuTime()/(double)Ntaus  << endl;

}//--> End of the macro



