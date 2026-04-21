#ifndef SCALINGSD_HH
#define SCALINGSD_HH

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "pipehit.hh"

class ScalingSD : public G4VSensitiveDetector{
	public:
		ScalingSD(const G4String& name);
		virtual ~ScalingSD() = default;

		virtual void Initialize(G4HCofThisEvent*) override;
		virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
		//virtual void EndOfEvent(G4HCofThisEvent*) override;
	

	private:
		PipeHitCollection* fHitsCollection;

};






#endif
