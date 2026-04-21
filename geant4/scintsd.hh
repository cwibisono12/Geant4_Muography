#ifndef SCINTSD_HH
#define SCINTSD_HH

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "scinthit.hh"

class ScintSD : public G4VSensitiveDetector{
	public:
		ScintSD(const G4String& name);
		virtual ~ScintSD() = default;

		virtual void Initialize(G4HCofThisEvent*) override;
		virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
		//virtual void EndOfEvent(G4HCofThisEvent*) override;
	

	private:
		ScintHitCollection* fHitsCollection;
		std::vector<G4bool> fHitsFlag;

};






#endif
