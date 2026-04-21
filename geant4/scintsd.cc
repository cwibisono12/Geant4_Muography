#include "scintsd.hh"
#include "G4SDManager.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
//#include "G4EventManager.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"

ScintSD::ScintSD(const G4String& name)
	:G4VSensitiveDetector(name), fHitsCollection(nullptr){
		collectionName.insert("ScintHitCollection");
	}

//PipeSD::~PipeSD(){}

void ScintSD::Initialize(G4HCofThisEvent* hce){
	fHitsFlag.assign(4,false);
	fHitsCollection = new ScintHitCollection(SensitiveDetectorName, collectionName[0]);
	//fHitsCollection = new PipeHitCollection(SensitiveDetectorName, "PipeHitCollection");

	G4int hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
	hce->AddHitsCollection(hcID, fHitsCollection);
//	G4cout << "DEBUG: Collection Created!" << G4endl;
}

G4bool ScintSD::ProcessHits(G4Step* step, G4TouchableHistory*){
		if (!fHitsCollection) return false;
	//	G4cout << "DEBUG: ProcessHits Started" << G4endl;
		G4Track* track = step->GetTrack();
		
		auto pd = track->GetDefinition();
	//	G4bool isMuon =	(track->GetDefinition() ==  G4MuonPlus::MuonPlusDefinition() ||
	//			track->GetDefinition() == G4MuonMinus::MuonMinusDefinition());

		G4bool isMuon = (pd == G4MuonPlus::Definition() || pd == G4MuonMinus::Definition());
		G4bool isPrimary = (track->GetParentID() == 0);

		G4int copyNo = step->GetPreStepPoint()->GetTouchableHandle()->GetCopyNumber();
		if (copyNo < 0 || copyNo >= 4) return false;
		//Only allow the first to the Pipe:
	//	if (fHitsCollection->entries() > 0) return false; //Comment out Apr 21 '26 as we allow more entries;
		if (fHitsFlag[copyNo]) return false;

		if (isMuon && isPrimary){
	//		G4cout << "DEBUG: Muon Filter Passed" << G4endl;
		//	G4int evtID = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
			ScintHit* newHit = new ScintHit();
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
			newHit->SetDetID(copyNo);
			newHit->SetPos(pos);

			fHitsCollection->insert(newHit);
			fHitsFlag[copyNo] = true;
	//		G4cout << "DEBUG: Hit Inserted" << G4endl;
			return true;
		}

		return false;
}

