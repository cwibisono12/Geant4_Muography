#include "action.hh"
#include "event.hh"

MyActionInitialization::MyActionInitialization(G4String fileName, G4int run_mode)
:G4VUserActionInitialization(), fname(fileName), mode(run_mode){}

MyActionInitialization::~MyActionInitialization()
{}

/* This is just for diagnostic:
void MyActionInitialization::Build() const {
SetUserAction(new MyPrimaryGenerator());
}
*/


// This is the old source code 'Oct 9/25
void MyActionInitialization::Build() const
{
   //Commented by C.W on Oct 15 '25
   //MyPrimaryGenerator *generator = new MyPrimaryGenerator();
   //SetUserAction(generator);
  
     SetUserAction(new MyPrimaryGenerator());

   //Commented by C.W on Oct 15 '25
 //  MyRunAction *runAction = new MyRunAction();
 //  SetUserAction(runAction);
     SetUserAction(new MyRunAction(fname));


//Added on Oct 28 '25 C.W     
  SetUserAction(new MyEventAction(mode));   

//SetUserAction(new SteppingAction());

}

