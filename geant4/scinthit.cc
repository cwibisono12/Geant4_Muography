#include "scinthit.hh"


ScintHit:: ScintHit()
	: G4VHit(),
	fPos(G4ThreeVector()),
	fTrackID(-1),
	fParentID(-1),
	fDetID(-1)

{}



ScintHit::~ScintHit()
{}


