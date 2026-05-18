#include "construction.hh"
#include "G4IntersectionSolid.hh"
#include "G4Transform3D.hh"
#include "G4SDManager.hh"

MyDetectorConstruction::MyDetectorConstruction(G4int run_mode)
	:G4VUserDetectorConstruction(), mode(run_mode)
{}

MyDetectorConstruction::~MyDetectorConstruction()
{}

G4VPhysicalVolume *MyDetectorConstruction::Construct()
{
  G4NistManager *nist = G4NistManager::Instance();
  G4Material *worldMat = nist->FindOrBuildMaterial("G4_AIR");

  G4Box *solidWorld = new G4Box("solidWorld", 1*m , 1*m, 1*m);
  G4LogicalVolume *logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld");
  G4VPhysicalVolume *physWorld = new G4PVPlacement(0, G4ThreeVector(0, 0, 0), logicWorld, "physWorld", 0, false, 0, true );

// define EcoMug plane-sky region
  G4Material *ecoMugMat = nist->FindOrBuildMaterial("G4_POLYSTYRENE");

  G4double planeSizeX = 2*m;
  G4double planeSizeY = 2*m;
  G4double planeThickness = 1*cm;

  G4Box *ecoMugSolid = new G4Box("ecoMugSolid", planeSizeX / 2, planeSizeY / 2, planeThickness / 2);
  G4LogicalVolume *ecoMugLogic = new G4LogicalVolume(ecoMugSolid, ecoMugMat, "ecoMugLogic");
  G4ThreeVector ecoMugPosition = G4ThreeVector(0, 0, 0.7*m);  // set to match EcoMug generator's center inside generator.cc
  //Use this line if we want to have more distance between top scintillators://Apr 20 '26
  //G4ThreeVector ecoMugPosition = G4ThreeVector(0, 0, 1.0*m);  // set to match EcoMug generator's center inside generator.cc
  new G4PVPlacement(0, ecoMugPosition, ecoMugLogic, "EcoMugPhys", logicWorld, false, 0, true);

  G4Material *FeC = new G4Material("FeC", 7.82*g/cm3, 2); // blacksteel
  FeC->AddElement(nist->FindOrBuildElement("Fe"), 1);   // Ferro mass fraction
  FeC->AddElement(nist->FindOrBuildElement("C"), 1);    // Carbon mass fraction

// Pipe Object
  G4double pipeInnerRadius = 18*cm;
  G4double pipeOuterRadius = 21*cm;
  G4double pipeLength = 60*cm;

  G4Tubs *solidPipe = new G4Tubs("solidPipe", pipeInnerRadius, pipeOuterRadius, pipeLength / 2, 0*deg, 360*deg);
  //G4LogicalVolume *logicPipe = new G4LogicalVolume(solidPipe, FeC, "logicPipe");
  logicPipe = new G4LogicalVolume(solidPipe, FeC, "logicPipe");
  G4ThreeVector pipePos = G4ThreeVector(0, 0, 0);

  G4RotationMatrix *rotation = new G4RotationMatrix();
  rotation->rotateX(90*deg); 

  new G4PVPlacement(rotation, pipePos, logicPipe, "physPipe", logicWorld, false, 0, true);

// creating silica amorf material with density 2.65 gr/cm3
  G4Material *SiO2 = new G4Material("SiO2", 2.65*g/cm3, 2); // silica amorf
  SiO2->AddElement(nist->FindOrBuildElement("Si"), 1);   // silicon mass fraction 46.7%
  SiO2->AddElement(nist->FindOrBuildElement("O"), 2);    // oxygen mass fraction 53.3%

// barium sulphate material
  G4Material *BaSO4 = new G4Material("BaSO4", 4.5*g/cm3, 3);
  BaSO4->AddElement(nist->FindOrBuildElement("Ba"), 1);
  BaSO4->AddElement(nist->FindOrBuildElement("S"), 1);
  BaSO4->AddElement(nist->FindOrBuildElement("O"), 4);

  G4double scalingInnerRadius = 0;
  G4double scalingOuterRadius = pipeInnerRadius;

// scaling 1/3 pipe
/*
  auto fullDisk = new G4Tubs("disk", scalingInnerRadius, scalingOuterRadius, pipeLength/2, 0*deg, 180*deg);
  auto cutter = new G4Box("cutter", scalingOuterRadius, scalingOuterRadius/2, pipeLength/2);
  auto cutterTransform = G4Translate3D(0, 17*cm, 0);
  auto solidScaling = new G4IntersectionSolid("solidScaling", fullDisk, cutter, cutterTransform);
  G4LogicalVolume *logicScaling = new G4LogicalVolume(solidScaling, BaSO4, "logicScaling");
  G4ThreeVector scalingPos = G4ThreeVector(0, 0, 0);

  new G4PVPlacement(rotation, scalingPos, logicScaling, "physScaling", logicWorld, false, 0, true);

*/
 G4double scalingR_in = 10*cm;
 G4double scalingR_out = 18*cm;
 G4Tubs *solidScaling = new G4Tubs("solidScaling", scalingR_in, scalingR_out, pipeLength/2, 0*deg, 360*deg);
 logicScaling = new G4LogicalVolume(solidScaling, BaSO4, "logicScaling");
 new G4PVPlacement(rotation, G4ThreeVector(0,0,0), logicScaling, "physScaling", logicWorld, false, 0, true);





// display each shape already defines
  //G4VisAttributes *pipeVisAtt = new G4VisAttributes(G4Colour(0.3, 0.3, 0.3, 0.5)); // abu
  G4VisAttributes *pipeVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 1.0)); // grey
  pipeVisAtt->SetVisibility(true);
  pipeVisAtt->SetForceSolid(true);
  logicPipe->SetVisAttributes(pipeVisAtt);

  //G4VisAttributes *ScalingVisAtt = new G4VisAttributes(G4Colour(1, 0, 1, 0.8)); // magenta
  G4VisAttributes *ScalingVisAtt = new G4VisAttributes(G4Colour(0., 0., 1, 0.8)); // blue
  ScalingVisAtt->SetVisibility(true);
  ScalingVisAtt->SetForceSolid(true);
  logicScaling->SetVisAttributes(ScalingVisAtt);

  G4VisAttributes *ecoMugVisAttr = new G4VisAttributes(G4Colour(0.9, 0.7, 0, 0.3));
  ecoMugVisAttr->SetVisibility(true);
  ecoMugVisAttr->SetForceSolid(true);
  ecoMugLogic->SetVisAttributes(ecoMugVisAttr);

// detector section
  G4Material *scintillator = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

  G4Box *solidDetector = new G4Box("solidDetector", 0.5*m, 0.01*cm, 0.5*m);
  logicDetector = new G4LogicalVolume(solidDetector, scintillator, "logicDetector");
//  G4RotationMatrix *rotation1 = new G4RotationMatrix();
//  rotation1->rotateX(90*deg);

  for (G4int i = 0; i < 4; i++)
  {
    if (i < 2)
    {
        G4VPhysicalVolume *physDetector = new G4PVPlacement(rotation,
            G4ThreeVector(0, 0, ((1 - i)*10*cm + 0.45*m)),
            //G4ThreeVector(0, 0, ((1 - i)*30*cm + 0.45*m)), //Apr 20 '26 (use this if we want to have more distance).
            logicDetector, "physDetector", logicWorld, false, i, true);
    }
    else
    {
        G4VPhysicalVolume *physDetector = new G4PVPlacement(rotation,
            G4ThreeVector(0, 0, -((i - 2)*10*cm + 0.45*m)),
            //G4ThreeVector(0, 0, -((i - 2)*30*cm + 0.45*m)), //Apr 20 '26 (use this if we want to have more distance)/
            logicDetector, "physDetector", logicWorld, false, i, true);
    }
  }

  return physWorld;
}

void MyDetectorConstruction::ConstructSDandField()
{
	/*
  MySensitiveDetector *sensDet = new MySensitiveDetector("SensitiveDetector");
  G4SDManager::GetSDMpointer()->AddNewDetector(sensDet);
  SetSensitiveDetector(logicDetector, sensDet);
  */
  // logicDetector->SetSensitiveDetector(sensDet);
 
 ScintSD* scintSD = new ScintSD("Scint_SensitiveDet");
 G4SDManager::GetSDMpointer()->AddNewDetector(scintSD);
 SetSensitiveDetector(logicDetector, scintSD);


 PipeSD* pipeSD = new PipeSD("Pipe_SensitiveDet", mode);
 G4SDManager::GetSDMpointer()->AddNewDetector(pipeSD);
 SetSensitiveDetector(logicPipe, pipeSD);
 //logicPipe->SetSensitiveDetector(pipeSD); 
 
 ScalingSD* scalingSD = new ScalingSD("Scaling_SensitiveDet", mode);
 G4SDManager::GetSDMpointer()->AddNewDetector(scalingSD);
 SetSensitiveDetector(logicScaling, scalingSD);
 
}
