#include "MyTrackingAction.hh"
#include "G4TrackingManager.hh"

void MyTrackingAction::PreUserTrackingAction(const G4Track* track) {
//G4cout << "Tracking particle:" << track->GetDefinition()->GetParticleName()
   //    << "parentID = " << track->GetParentID() << G4endl;
    // Keep only primary tracks
    if (track->GetParentID() == 0)
        fpTrackingManager->SetStoreTrajectory(true);
    else
        fpTrackingManager->SetStoreTrajectory(false);
}

