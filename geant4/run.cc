#include "run.hh"
//#include <fstream>
#include <string>
#include "G4AnalysisManager.hh"
#include "G4Run.hh"

MyRunAction::MyRunAction(G4String fileName, G4int mode) : G4UserRunAction(), fname(fileName), run_mode(mode){
	auto analysisManager = G4AnalysisManager::Instance();
	analysisManager->SetDefaultFileType("csv");

	//Output File0: Scintillator Hits:
	analysisManager->CreateNtuple("ScintillatorHits","Primary Muon Data");
	analysisManager->CreateNtupleIColumn("EventID");
	analysisManager->CreateNtupleIColumn("ParentID");
	analysisManager->CreateNtupleIColumn("TrackID");
	analysisManager->CreateNtupleIColumn("detID");
	analysisManager->CreateNtupleDColumn("X_mm");
	analysisManager->CreateNtupleDColumn("Y_mm");
	analysisManager->CreateNtupleDColumn("Z_mm");
	analysisManager->FinishNtuple();

	//Output File1: PipeHit
	analysisManager->CreateNtuple("PipeHits","Primary Muon Data");
	analysisManager->CreateNtupleIColumn("EventID");
	analysisManager->CreateNtupleIColumn("ParentID");
	analysisManager->CreateNtupleIColumn("TrackID");
	analysisManager->CreateNtupleDColumn("X_mm");
	analysisManager->CreateNtupleDColumn("Y_mm");
	analysisManager->CreateNtupleDColumn("Z_mm");
	analysisManager->FinishNtuple();

	//Output File1: ScalingHit
	analysisManager->CreateNtuple("ScalingHits","Primary Muon Data");
	analysisManager->CreateNtupleIColumn("EventID");
	analysisManager->CreateNtupleIColumn("ParentID");
	analysisManager->CreateNtupleIColumn("TrackID");
	analysisManager->CreateNtupleDColumn("X_mm");
	analysisManager->CreateNtupleDColumn("Y_mm");
	analysisManager->CreateNtupleDColumn("Z_mm");
	analysisManager->FinishNtuple();
}

MyRunAction::~MyRunAction()
{}

void MyRunAction::BeginOfRunAction(const G4Run*) {
    auto analysisManager = G4AnalysisManager::Instance();
   fname = fname + "_" + std::to_string(run_mode);
   analysisManager->OpenFile(fname);
}


void MyRunAction::EndOfRunAction(const G4Run*){
	auto analysisManager = G4AnalysisManager::Instance();

	analysisManager->Write();
	analysisManager->CloseFile();
}
