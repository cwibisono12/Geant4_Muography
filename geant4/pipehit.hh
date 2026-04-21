#ifndef PIPEHIT_HH
#define PIPEHIT_HH

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"
#include "G4Allocator.hh"

class PipeHit : public G4VHit{
	public:
		PipeHit();
		virtual ~PipeHit();

		void SetEventID(G4int id){fEventID = id;}
		void SetPos(G4ThreeVector pos) {fPos = pos;}
		void SetTrackID(G4int id) {fTrackID = id;}
		void SetParentID(G4int id){fParentID = id;}


		G4ThreeVector GetPos() const{return fPos;}
		G4int GetTrackID() const {return fTrackID;}
		G4int GetParentID() const {return fParentID;}
		G4int GetEventID() const {return fEventID;}

	private:
		G4int fEventID;
		G4ThreeVector fPos;
		G4int	fTrackID;
		G4int fParentID;
		
};


typedef G4THitsCollection<PipeHit> PipeHitCollection;
//extern G4ThreadLocal G4Allocator<PipeHit> * PipeHitAllocator;


#endif
