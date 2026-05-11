#ifndef ACTION_HH
#define ACTION_HH

#include "G4VUserActionInitialization.hh"

#include "generator.hh"
#include "run.hh"

class MyActionInitialization : public G4VUserActionInitialization
{
public:
   MyActionInitialization(G4String fileName, G4int run_mode);
   ~MyActionInitialization();

   void Build() const override;
   G4String fname;
   G4int mode;
};

#endif
