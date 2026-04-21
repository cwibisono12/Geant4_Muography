#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4MuonMinus.hh" 
#include "G4MuonPlus.hh" 
#include "Randomize.hh"

#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4ParticleDefinition.hh" //added by CW Oct 13'25
#include "EcoMug.h"

class MyPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
public:
	MyPrimaryGenerator();
	~MyPrimaryGenerator();
	
	virtual void GeneratePrimaries(G4Event*);
private:
	//generate particle gun
	G4ParticleGun *fParticleGun;
        G4ParticleDefinition *mu_plus, *mu_minus; //Uncomment Oct 9 '25
	EcoMug fMuonGen;	
};

#endif
