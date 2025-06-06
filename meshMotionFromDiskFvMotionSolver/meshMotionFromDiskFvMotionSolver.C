/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2017 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "meshMotionFromDiskFvMotionSolver.H"
#include "motionInterpolation.H"
#include "motionDiffusivity.H"
#include "fvmLaplacian.H"
#include "addToRunTimeSelectionTable.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "surfaceInterpolate.H"
#include "fvcLaplacian.H"
#include "mapPolyMesh.H"
#include "fvOptions.H"
#include "IFstream.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(meshMotionFromDiskFvMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        meshMotionFromDiskFvMotionSolver,
        dictionary
    );

    addToRunTimeSelectionTable
    (
        displacementMotionSolver,
        meshMotionFromDiskFvMotionSolver,
        displacement
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::meshMotionFromDiskFvMotionSolver::meshMotionFromDiskFvMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict
)
:
    displacementMotionSolver(mesh, dict, typeName),
    fvMotionSolver(mesh),
    dispFilesDirectory_(motionSolver::coeffDict().lookup("dispFilesDirectory"))
{}


Foam::meshMotionFromDiskFvMotionSolver::meshMotionFromDiskFvMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict,
    const pointVectorField& pointDisplacement,
    const pointIOField& points0
)
:
    displacementMotionSolver(mesh, dict, pointDisplacement, points0, typeName),
    fvMotionSolver(mesh),
    dispFilesDirectory_(motionSolver::coeffDict().lookup("dispFilesDirectory"))
{
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::meshMotionFromDiskFvMotionSolver::
~meshMotionFromDiskFvMotionSolver()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField>
Foam::meshMotionFromDiskFvMotionSolver::curPoints() const
{
    tmp<pointField> tcurPoints
    (
        points0() + pointDisplacement_.primitiveField()
    );

    twoDCorrectPoints(tcurPoints.ref());

    return tcurPoints;
}


void Foam::meshMotionFromDiskFvMotionSolver::solve()
{
    // The points have moved so before interpolation update
    // the motionSolver accordingly
    movePoints(fvMesh_.points());

    // Read the displacement field for this time index
    const fileName dispFileName
    (
        fileName(dispFilesDirectory_/"pointDisplacement_")
       + Foam::name(time().timeIndex())
    );
    Info<< "Reading " << dispFileName << endl;
    IFstream dispFile(dispFileName);

    // Read the point displacement field from the dispFile
    const vectorField disp(dispFile);

    if (disp.size() != pointDisplacement_.primitiveField().size())
    {
        FatalErrorInFunction
            << "The size of the field in " << dispFileName << " is "
            << disp.size() << ", but it should be "
            << pointDisplacement_.primitiveField().size() << abort(FatalError);
    }

    // Assign the disp field to the pointDisplacement field and correct the
    // boundary conditions
    pointDisplacement_.primitiveFieldRef() = disp;
    pointDisplacement_.correctBoundaryConditions();
}


void Foam::meshMotionFromDiskFvMotionSolver::updateMesh
(
    const mapPolyMesh& mpm
)
{
    displacementMotionSolver::updateMesh(mpm);
}


// ************************************************************************* //
