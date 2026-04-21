#ifndef SCINTHIT_HH
#define SCINTHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"
#include "G4Allocator.hh"

class ScintHit : public G4VHit{
	public:
		ScintHit();
		virtual ~ScintHit();

		void SetEventID(G4int id){fEventID = id;}
		void SetPos(G4ThreeVector pos) {fPos = pos;}
		void SetTrackID(G4int id) {fTrackID = id;}
		void SetParentID(G4int id){fParentID = id;}
		void SetDetID(G4int id){fDetID = id;} 

		G4ThreeVector GetPos() const{return fPos;}
		G4int GetTrackID() const {return fTrackID;}
		G4int GetParentID() const {return fParentID;}
		G4int GetEventID() const {return fEventID;}
		G4int GetDetID() const {return fDetID;}

	private:
		G4int fEventID;
		G4ThreeVector fPos;
		G4int	fTrackID;
		G4int fParentID;
		G4int fDetID;
		
};


typedef G4THitsCollection<ScintHit> ScintHitCollection;
//extern G4ThreadLocal G4Allocator<PipeHit> * PipeHitAllocator;


#endif
