#ifndef EVENT_HH
#define EVENT_HH

#include "G4UserEventAction.hh"
#include "globals.hh"
#include "G4Event.hh"
#include <fstream>

class MyEventAction : public G4UserEventAction
{
public:
  MyEventAction();
  virtual ~MyEventAction();

  void BeginOfEventAction(const G4Event* event) override;
  void EndOfEventAction(const G4Event* event) override;
private:
  // Additional member variables, if needed.
  //std::ofstream fEbeamlog;
  G4int fHCID0;
  G4int fHCID1;
  G4int fHCID2;
};

#endif
