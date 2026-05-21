#include "pipesd.hh"
#include "G4SDManager.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
//#include "G4EventManager.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"

PipeSD::PipeSD(const G4String& name, G4int run_mode)
	:G4VSensitiveDetector(name), fHitsCollection(nullptr), mode(run_mode){
		collectionName.insert("PipeHitCollection");
	}

//PipeSD::~PipeSD(){}

void PipeSD::Initialize(G4HCofThisEvent* hce){
	fHitsCollection = new PipeHitCollection(SensitiveDetectorName, collectionName[0]);
	//fHitsCollection = new PipeHitCollection(SensitiveDetectorName, "PipeHitCollection");
	//Only store the last hit for this mode:


	G4int hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
	hce->AddHitsCollection(hcID, fHitsCollection);
//	G4cout << "DEBUG: Collection Created!" << G4endl;
}

G4bool PipeSD::ProcessHits(G4Step* step, G4TouchableHistory*){
		if (!fHitsCollection) return false;
	//	G4cout << "DEBUG: ProcessHits Started" << G4endl;
		G4Track* track = step->GetTrack();
		
		auto pd = track->GetDefinition();
	//	G4bool isMuon =	(track->GetDefinition() ==  G4MuonPlus::MuonPlusDefinition() ||
	//			track->GetDefinition() == G4MuonMinus::MuonMinusDefinition());

		G4bool isMuon = (pd == G4MuonPlus::Definition() || pd == G4MuonMinus::Definition());
		G4bool isPrimary = (track->GetParentID() == 0);

		//Only allow the first to the Pipe:
		if (mode == 1){
		if (fHitsCollection->entries() > 0) return false;
		if (isMuon && isPrimary){
	//		G4cout << "DEBUG: Muon Filter Passed" << G4endl;
		//	G4int evtID = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
			PipeHit* newHit = new PipeHit();
	//		G4cout << "DEBUG: Hit Object Created" << G4endl;

			G4int evtID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
			if(!evtID) {
				G4cout << "DEBUG: EVENT POINTER IS NULL!" << G4endl; return false;
			}



			newHit->SetEventID(evtID);
			G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();
			
			G4int trackID = step->GetTrack()->GetTrackID(); //Sanity check  = 1 invalid otherwise;
			G4int parentID = step->GetTrack()->GetParentID(); //Sanity check = 0 invalid otherwise;
			newHit->SetTrackID(trackID);
			newHit->SetParentID(parentID);
			newHit->SetPos(pos);

			fHitsCollection->insert(newHit);
	//		G4cout << "DEBUG: Hit Inserted" << G4endl;
			return true;
			}
		}

		//Only allow the last hit of the Pipe
		if (mode == 2){
		if (isMuon && isPrimary){
	//		G4cout << "DEBUG: Muon Filter Passed" << G4endl;
		//	G4int evtID = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
			for(G4int i=0; i<fHitsCollection->entries();i++){
				delete (*fHitsCollection)[i];				
			}
			fHitsCollection->GetVector()->clear();

			PipeHit* newHit = new PipeHit();
	//		G4cout << "DEBUG: Hit Object Created" << G4endl;

			G4int evtID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
			if(!evtID) {
				G4cout << "DEBUG: EVENT POINTER IS NULL!" << G4endl; return false;
			}



			newHit->SetEventID(evtID);
			G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();
			
			G4int trackID = step->GetTrack()->GetTrackID(); //Sanity check  = 1 invalid otherwise;
			G4int parentID = step->GetTrack()->GetParentID(); //Sanity check = 0 invalid otherwise;
			newHit->SetTrackID(trackID);
			newHit->SetParentID(parentID);
			newHit->SetPos(pos);

			fHitsCollection->insert(newHit);
	//		G4cout << "DEBUG: Hit Inserted" << G4endl;
			return true;
			}
		}

		//Store the entire hits as the particle passes through the object
		if (mode == 3 || mode == 4 || mode == 5){
		if (isMuon && isPrimary){
	//		G4cout << "DEBUG: Muon Filter Passed" << G4endl;
		//	G4int evtID = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
			PipeHit* newHit = new PipeHit();
	//		G4cout << "DEBUG: Hit Object Created" << G4endl;

			G4int evtID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
			if(!evtID) {
				G4cout << "DEBUG: EVENT POINTER IS NULL!" << G4endl; return false;
			}



			newHit->SetEventID(evtID);
			G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();
			
			G4int trackID = step->GetTrack()->GetTrackID(); //Sanity check  = 1 invalid otherwise;
			G4int parentID = step->GetTrack()->GetParentID(); //Sanity check = 0 invalid otherwise;
			newHit->SetTrackID(trackID);
			newHit->SetParentID(parentID);
			newHit->SetPos(pos);

			fHitsCollection->insert(newHit);
	//		G4cout << "DEBUG: Hit Inserted" << G4endl;
			return true;
			}
		}

		//Only take the average as the particle passes through the object

		return false;
}

