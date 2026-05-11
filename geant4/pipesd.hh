#ifndef PIPESD_HH
#define PIPESD_HH

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "pipehit.hh"

class PipeSD : public G4VSensitiveDetector{
	public:
		PipeSD(const G4String& name, G4int run_mode);
		virtual ~PipeSD() = default;

		virtual void Initialize(G4HCofThisEvent*) override;
		virtual G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
		//virtual void EndOfEvent(G4HCofThisEvent*) override;
	

	private:
		PipeHitCollection* fHitsCollection;
		G4int mode; //run mode to choose the hits being recorded

};






#endif
