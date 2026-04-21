#include "EcoMug.h"
#include "generator.hh"
#include "G4ParticleGun.hh" //added on Oct 9 '25
#include "G4ParticleTable.hh" //added on Oct 9 '25
#include "G4ParticleDefinition.hh"
//#include <fstream> //Commented out Oct 28 '25 C.W

/* This is just for diagnostic
MyPrimaryGenerator::MyPrimaryGenerator()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(0)
{

    fParticleGun = new G4ParticleGun(1);
    auto table = G4ParticleTable::GetParticleTable();
    auto particle = table->FindParticle("geantino");
    if(!particle) G4Exception("MyPrimaryGenerator","NoParticle",FatalException,"geantino not found");
    fParticleGun->SetParticleDefinition(particle);
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,1.));
    fParticleGun->SetParticleEnergy(1.0*GeV);

   

}
*/

//This is the actual constructor:

MyPrimaryGenerator::MyPrimaryGenerator()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(0), mu_plus(nullptr), mu_minus(nullptr)
{
//commented out by C.W Oct '13/25
   // std::ofstream clear("muon_energy_log.csv", std::ios::out);
   // clear.close();

    fMuonGen.SetUseSky();
    fMuonGen.SetSkySize({{2*m, 2*m}});
    fMuonGen.SetSkyCenterPosition({{0, 0, 0.7*m}});
    //fMuonGen.SetSkyCenterPosition({{0, 0, 0.7*m}}); //Apr 20 '26 use this if we want to have more distance

//    fMuonGen.SetSeed(1234); //Comment this line out if we want to have the same results everytime we run

    fParticleGun = new G4ParticleGun(1);
//    mu_minus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
//    mu_plus = G4ParticleTable::GetParticleTable()->FindParticle("mu+");

//Added this one to prevent segmentation fault 10/20 '25 CW
/*
   if(!mu_minus){
G4Exception("PrimaryGen","NoParticle",FatalException,"mu- not found");
}
   if(!mu_plus){
G4Exception("PrimaryGen","NoParticle",FatalException,"mu+ not found");
}
*/


}


MyPrimaryGenerator::~MyPrimaryGenerator()
{
    delete fParticleGun;
}

// This function is just for diagnostic:
/*
void MyPrimaryGenerator::GeneratePrimaries(G4Event *anEvent){

fParticleGun->GeneratePrimaryVertex(anEvent);
}
*/

// original source code
void MyPrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    fMuonGen.Generate();

int muon_type = fMuonGen.GetCharge();
    mu_minus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
    mu_plus = G4ParticleTable::GetParticleTable()->FindParticle("mu+");
if(!mu_plus || !mu_minus){
G4Exception("MyPrimaryGenerator::MyPrimaryGenerator()","MissingParticle",
	   FatalException,"Could not find mu+ or mu- in G4ParticleTable."
	   "Check your physics list.");
}
    if (muon_type == -1) {
	 fParticleGun->SetParticleDefinition(mu_minus);
	printf("mu_minus\n");
    } 
    if (muon_type == 1){
       fParticleGun->SetParticleDefinition(mu_plus);
	printf("mu_plus\n");
    }


    // obtain the muon generation parameters
    std::array<double, 3> muon_pos = fMuonGen.GetGenerationPosition();
    double muon_ptot = fMuonGen.GetGenerationMomentum();
    double muon_theta = fMuonGen.GetGenerationTheta();
    double muon_phi = fMuonGen.GetGenerationPhi();

    // set the particle's position
    fParticleGun->SetParticlePosition(G4ThreeVector(
	muon_pos[0]*mm,
	muon_pos[1]*mm,
	muon_pos[2]*mm));

    // set the particle's momentum direction and magnitude
    fParticleGun->SetParticleMomentum(
        G4ParticleMomentum(muon_ptot * sin(muon_theta) * cos(muon_phi) * GeV,
                           muon_ptot * sin(muon_theta) * sin(muon_phi) * GeV,
                           muon_ptot * cos(muon_theta) * GeV));

    // set the particle definition based on charge


//printf("charged: %d ",fMuonGen.GetCharge());

//Test Checked whether fParticleGun object can be feed by G4ParticleDefinition default particle such as e-
//G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");
//fParticleGun->SetParticleDefinition(particle);

//G4cout <<"Found particle: " << mu_plus->GetParticleName() << G4endl;
//G4cout <<"Found particle: " << mu_minus->GetParticleName() << G4endl;

//if(mu_plus && muon_type == 1){printf("mu_plus\n"); }//fParticleGun->SetParticleDefinition(mu_plus);
//if(mu_minus && muon_type == -1){printf("mu_minus\n");} //fParticleGun->SetParticleDefinition(mu_minus);
//fParticleGun->SetParticleDefinition(mu_plus);
/* Particle definition commented by C.W Oct '13/25
    fParticleGun->SetParticleDefinition(mu_minus);
    fParticleGun->SetParticleDefinition(mu_plus);
*/
    

//Print the energy Oct 28/'25 C.W
//G4cout << ">>> Gun set energy = "
//<< fParticleGun->GetParticleEnergy() << G4endl;

    // generate the primary vertex
    fParticleGun->GeneratePrimaryVertex(anEvent);
//    G4cout << "muon_ptot = " << muon_ptot << G4endl;

    // Hitung energi kinetik dari momentum
/*
    double mass = 0.105658; // massa muon dalam GeV/c²
    double energy_total = std::sqrt(muon_ptot * muon_ptot + mass * mass); // E = sqrt(p² + m²)
    double kinetic_energy = energy_total - mass; // E_kin = E_total - m
    G4cout << "Generated muon with:"
       << " p = " << muon_ptot << " GeV/c"
       << ", theta = " << muon_theta << " rad"
       << ", phi = " << muon_phi << " rad"
       << ", E_kin = " << kinetic_energy << " MeV" // MeV
       << G4endl;
*/
 

   //These section is commented out by C.W Oct 28/'25 
   //The purpose is to have an event ID printed so that it can be retrieved with the hits as well.
   /*
   double mass = 0.105658; // GeV
    double energy_total = std::sqrt(muon_ptot * muon_ptot + mass * mass);
    double kinetic_energy = energy_total - mass; // dalam GeV

    std::ofstream file("muon_energy_log.csv", std::ios::app);
    file << kinetic_energy << "\n"; // GeV
    file.close();

    */
    
    }

