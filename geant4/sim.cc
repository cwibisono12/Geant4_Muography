#include <iostream>
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "construction.hh"
#include "FTFP_BERT.hh"
#include "G4StepLimiterPhysics.hh"
#include "action.hh"

int main(int argc, char** argv)
{
   G4String filename = argv[1];   
   G4int beamnumber = atoi(argv[2]);
   G4int run_mode = atoi(argv[3]); //mode to store the hits for the pipe and scaling
   G4RunManager *runManager = new G4RunManager();
   
   runManager->SetUserInitialization(new MyDetectorConstruction(run_mode));
   auto MyPhysicsList = new FTFP_BERT;
   MyPhysicsList->RegisterPhysics(new G4StepLimiterPhysics());
   runManager->SetUserInitialization(MyPhysicsList);
   runManager->SetUserInitialization(new MyActionInitialization(filename, run_mode) );

  
//Commented to correct the order by C.W Oct 13/'25
   runManager->Initialize();
   
 /*
   G4VisManager *visManager = new G4VisExecutive();
   visManager->Initialize();
      
   G4UIExecutive *ui = new G4UIExecutive(argc, argv); 
   G4UImanager *UImanager= G4UImanager::GetUIpointer();
   UImanager->ApplyCommand("/vis/open OGL");
   UImanager->ApplyCommand("/vis/viewer/set/viewpointVector 1 1 1");
   UImanager->ApplyCommand("/vis/drawVolume");
   UImanager->ApplyCommand("/vis/ogl/set/displayListLimit 1000000");
 
*/

 /*
   UImanager->ApplyCommand("/vis/scene/add/trajectories");
   UImanager->ApplyCommand("/vis/modeling/trajectories/create/drawByParticleID");
   UImanager->ApplyCommand("/vis/filter/trajectories/create/particleFilter");
   UImanager->ApplyCommand("/vis/filter/trajectories/particleFilter-0/add mu+");
   UImanager->ApplyCommand("/vis/filter/trajectories/particleFilter-0/add mu-");
*/

 /* 
   UImanager->ApplyCommand("/vis/scene/add/trajectories smooth"); //Comment by C.W Oct 13/'25
   UImanager->ApplyCommand("/vis/scene/add/hits"); //added by C.W on Oct 15 '25
   UImanager->ApplyCommand("/vis/viewer/set/autoRefresh true");
   UImanager->ApplyCommand("/vis/scene/endOfEventAction accumulate");
*/
   runManager->BeamOn(beamnumber);
   //runManager->BeamOn(5);
/*
   ui->SessionStart();
   delete visManager;
   delete runManager;
   delete ui;
 */
 return 0;
}
