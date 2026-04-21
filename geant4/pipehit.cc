#include "pipehit.hh"


PipeHit:: PipeHit()
	: G4VHit(),
	fPos(G4ThreeVector()),
	fTrackID(-1),
	fParentID(-1)

{}



PipeHit::~PipeHit()
{}


