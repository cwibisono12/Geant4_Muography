#include "run.hh"
#include <fstream>
#include "G4AnalysisManager.hh"
#include "G4Run.hh"

MyRunAction::MyRunAction(G4String fileName) : G4UserRunAction(), fname(fileName){
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
 /*
    std::ofstream hitsfile("detector_hits.csv");
    hitsfile << "#event, parentID, trackID, detNum, x, y, z\n";
    hitsfile.close();

    std::ofstream mscfile("msc_points.csv");
    mscfile << "#event, parentID, trackID, x, y, z\n";
    mscfile.close();
*/
    auto analysisManager = G4AnalysisManager::Instance();
    //analysisManager->OpenFile("Obj_hits.csv");
    analysisManager->OpenFile(fname);
}


void MyRunAction::EndOfRunAction(const G4Run*){
	auto analysisManager = G4AnalysisManager::Instance();

	analysisManager->Write();
	analysisManager->CloseFile();
}
