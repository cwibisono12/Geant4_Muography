#include "event.hh"
#include "pipehit.hh" //Newly added 04/09 '26
#include "scinthit.hh" //Newly added 04/21 '26
#include "G4SDManager.hh" //Newly added 04/09 '26
#include "G4AnalysisManager.hh" //Newly added 04/09 '26

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4RunManager.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include <fstream>

MyEventAction::MyEventAction(G4int run_mode): G4UserEventAction(), fHCID0(-1), fHCID1(-1), fHCID2(-1), mode(run_mode){
//fEbeamlog.open("beam_energy.csv");
//fEbeamlog << "#EventID,BeamKineticEnergy_GeV"<< std::endl;
}

MyEventAction::~MyEventAction(){
//if (fEbeamlog.is_open()) fEbeamlog.close();
}

//Get the primary vertex
void MyEventAction::BeginOfEventAction(const G4Event* event){
const G4PrimaryVertex* vertex = event->GetPrimaryVertex();
if (!vertex) return;

//Get the primary particle from the vertex
const G4PrimaryParticle* primary = vertex->GetPrimary();
if (!primary) return;

//Retrieve Kinetic Energy
G4double kineticEnergy = primary->GetKineticEnergy();


//Convert to GeV:
G4double E_beam = kineticEnergy / GeV;

//Write to CSV:
//fEbeamlog << event->GetEventID() << ","
//	<< E_beam << std::endl;


G4cout << ">>> Event" <<event->GetEventID()
	<< " : Primary beam kinetic energy = "
	<< G4BestUnit(kineticEnergy, "Energy") << G4endl;

}


void MyEventAction::EndOfEventAction(const G4Event* event){
	G4cout << "DEBUG: Event End Start" << G4endl;
	G4HCofThisEvent* hce = event->GetHCofThisEvent();
	if(!hce) {
			G4cout << "DEBUG: No HCE" << G4endl;
			return;
	}
	auto sdManager = G4SDManager::GetSDMpointer();
	//G4cout << "Hit Recorded" << G4endl;
	if (fHCID0 == -1){
		fHCID0 = sdManager->GetCollectionID("ScintHitCollection");
		G4cout << "DEBUG: Found ID: " << fHCID0 << G4endl;
	}

	

	if (fHCID0 < 0) {
		G4cerr << "ERROR: ScintHitCollectionID not found! Check your SD constructor!" << G4endl;
		return;
	}
	auto hitCollection0 = static_cast<ScintHitCollection*>(hce->GetHC(fHCID0));
//	G4cout << "DEBUG: Collection Pointer: " << hitCollection << G4endl;

	if(hitCollection0 && hitCollection0->entries() > 0){
		auto analysisManager = G4AnalysisManager::Instance();
		//G4int eventID = event->GetEventID();
		for(G4int i = 0;i < hitCollection0->entries(); i++){
//			G4cout << "DEBUG: Entries: " << hitCollection->entries() << G4endl;
			ScintHit* hit = (*hitCollection0)[i];

			if(!hit) {
					G4cout << "DEBUG: Hit is NULL" << G4endl; 
					continue;
			}

			analysisManager->FillNtupleIColumn(0,0,hit->GetEventID());
			analysisManager->FillNtupleIColumn(0,1,hit->GetParentID());
			analysisManager->FillNtupleIColumn(0,2,hit->GetTrackID());
			analysisManager->FillNtupleIColumn(0,3,hit->GetDetID());
			analysisManager->FillNtupleDColumn(0,4,hit->GetPos().x());
			analysisManager->FillNtupleDColumn(0,5,hit->GetPos().y());
			analysisManager->FillNtupleDColumn(0,6,hit->GetPos().z());
		
			analysisManager->AddNtupleRow(0);

		}
//		G4cout << "DEBUG: Event End Finished" << G4endl;

	}



	if (fHCID1 == -1){
		fHCID1 = sdManager->GetCollectionID("PipeHitCollection");
		G4cout << "DEBUG: Found ID: " << fHCID1 << G4endl;
	}

	

	if (fHCID1 < 0) {
		G4cerr << "ERROR: PipeHitCollectionID not found! Check your SD constructor!" << G4endl;
		return;
	}
	auto hitCollection1 = static_cast<PipeHitCollection*>(hce->GetHC(fHCID1));
//	G4cout << "DEBUG: Collection Pointer: " << hitCollection << G4endl;

	if(hitCollection1 && hitCollection1->entries() > 0){
		auto analysisManager = G4AnalysisManager::Instance();
		//G4int eventID = event->GetEventID();

		if(mode == 1 || mode == 2 || mode == 3){
		for(G4int i = 0;i < hitCollection1->entries(); i++){
//			G4cout << "DEBUG: Entries: " << hitCollection->entries() << G4endl;
			PipeHit* hit = (*hitCollection1)[i];

			if(!hit) {
					G4cout << "DEBUG: Hit is NULL" << G4endl; 
					continue;
			}

			analysisManager->FillNtupleIColumn(1,0,hit->GetEventID());
			analysisManager->FillNtupleIColumn(1,1,hit->GetParentID());
			analysisManager->FillNtupleIColumn(1,2,hit->GetTrackID());
			analysisManager->FillNtupleDColumn(1,3,hit->GetPos().x());
			analysisManager->FillNtupleDColumn(1,4,hit->GetPos().y());
			analysisManager->FillNtupleDColumn(1,5,hit->GetPos().z());
		
			analysisManager->AddNtupleRow(1);

			}
//		G4cout << "DEBUG: Event End Finished" << G4endl;
		}
		if(mode == 4){
			G4double aveX = 0;
			G4double aveY = 0;
			G4double aveZ = 0;
			G4int evID, parentID, trackID;
		for(G4int i = 0;i < hitCollection1->entries(); i++){
//			G4cout << "DEBUG: Entries: " << hitCollection->entries() << G4endl;
			PipeHit* hit = (*hitCollection1)[i];
			
			if(!hit) {
					G4cout << "DEBUG: Hit is NULL" << G4endl; 
					continue;
			}
			aveX = aveX + hit->GetPos().x();
			aveY = aveY + hit->GetPos().y();
			aveZ = aveZ+ hit->GetPos().z();
			evID = hit->GetEventID();
			parentID = hit->GetParentID();
			trackID = hit->GetTrackID();


			}

			aveX = aveX/hitCollection1->entries();
			aveY = aveY/hitCollection1->entries();
			aveZ = aveZ/hitCollection1->entries();

			analysisManager->FillNtupleIColumn(1,0,evID);
			analysisManager->FillNtupleIColumn(1,1,parentID);
			analysisManager->FillNtupleIColumn(1,2,trackID);
			analysisManager->FillNtupleDColumn(1,3,aveX);
			analysisManager->FillNtupleDColumn(1,4,aveY);
			analysisManager->FillNtupleDColumn(1,5,aveZ);
		
			analysisManager->AddNtupleRow(1);

		}

	}



	if (fHCID2 == -1){
		fHCID2 = sdManager->GetCollectionID("ScalingHitCollection");
		G4cout << "DEBUG: Found ID: " << fHCID2 << G4endl;
	}

	

	if (fHCID2 < 0) {
		G4cerr << "ERROR: PipeHitCollectionID not found! Check your SD constructor!" << G4endl;
		return;
	}
	auto hitCollection2 = static_cast<PipeHitCollection*>(hce->GetHC(fHCID2));
//	G4cout << "DEBUG: Collection Pointer: " << hitCollection << G4endl;

	if(hitCollection2 && hitCollection2->entries() > 0){
		auto analysisManager = G4AnalysisManager::Instance();
		//G4int eventID = event->GetEventID();
		if(mode == 1 || mode == 2 || mode == 3){
		for(G4int i = 0;i < hitCollection2->entries(); i++){
//			G4cout << "DEBUG: Entries: " << hitCollection->entries() << G4endl;
			PipeHit* hit = (*hitCollection2)[i];

			if(!hit) {
					G4cout << "DEBUG: Hit is NULL" << G4endl; 
					continue;
			}

			analysisManager->FillNtupleIColumn(2,0,hit->GetEventID());
			analysisManager->FillNtupleIColumn(2,1,hit->GetParentID());
			analysisManager->FillNtupleIColumn(2,2,hit->GetTrackID());
			analysisManager->FillNtupleDColumn(2,3,hit->GetPos().x());
			analysisManager->FillNtupleDColumn(2,4,hit->GetPos().y());
			analysisManager->FillNtupleDColumn(2,5,hit->GetPos().z());
		
			analysisManager->AddNtupleRow(2);

		}
//		G4cout << "DEBUG: Event End Finished" << G4endl;

		}

		if(mode == 4){
		G4double aveX = 0;
		G4double aveY = 0;
		G4double aveZ = 0;
		G4int evID, parentID, trackID;

		for(G4int i = 0;i < hitCollection2->entries(); i++){
//			G4cout << "DEBUG: Entries: " << hitCollection->entries() << G4endl;
			PipeHit* hit = (*hitCollection2)[i];

			if(!hit) {
					G4cout << "DEBUG: Hit is NULL" << G4endl; 
					continue;
			}
			aveX = aveX + hit->GetPos().x();
			aveY = aveY + hit->GetPos().y();
			aveZ = aveZ + hit->GetPos().z();
			evID = hit->GetEventID();
			parentID = hit->GetParentID();
			trackID = hit->GetTrackID();
			}
			
			aveX = aveX / hitCollection2->entries();
			aveY = aveY / hitCollection2->entries();
			aveZ = aveZ / hitCollection2->entries();

			analysisManager->FillNtupleIColumn(2,0,evID);
			analysisManager->FillNtupleIColumn(2,1,parentID);
			analysisManager->FillNtupleIColumn(2,2,trackID);
			analysisManager->FillNtupleDColumn(2,3,aveX);
			analysisManager->FillNtupleDColumn(2,4,aveY);
			analysisManager->FillNtupleDColumn(2,5,aveZ);
		
			analysisManager->AddNtupleRow(2);

		}

	}
}

/*
void MyEventAction::EndOfEventAction(const G4Event* event)
{
  // Check if the event has primary vertices
  if (event->GetNumberOfPrimaryVertex() == 0) {
    G4cout << "Event has no primary vertices." << G4endl;
    return;
  }

  // Get the first primary vertex (assuming a single primary particle)
  G4PrimaryVertex* primaryVertex = event->GetPrimaryVertex(0);
  if (!primaryVertex) {
    return;
  }

  // Get the number of primary particles from this vertex
  G4int numParticles = primaryVertex->GetNumberOfParticle();
  if (numParticles == 0) {
    G4cout << "Primary vertex has no particles." << G4endl;
    return;
  }

  // Get the first primary particle from the vertex
  G4PrimaryParticle* primaryParticle = primaryVertex->GetPrimary(0);

  // Retrieve the kinetic energy
  if (primaryParticle) {
    G4double kineticEnergy = primaryParticle->GetKineticEnergy();

    // Print the result (with unit conversion for readability)
    G4cout << "Beam kinetic energy: " << kineticEnergy / MeV << " MeV" << G4endl;
  }
}
*/
