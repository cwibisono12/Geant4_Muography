#include "G4UserTrackingAction.hh"
#include "globals.hh"

class MyTrackingAction : public G4UserTrackingAction {
public:
    virtual void PreUserTrackingAction(const G4Track* track) override;
};

