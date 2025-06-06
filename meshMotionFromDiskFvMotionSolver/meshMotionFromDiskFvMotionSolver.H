/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::meshMotionFromDiskFvMotionSolver

Group
    grpMeshMotionSolvers

Description
    Mesh motion solver for an fvMesh where the point displacement field is read
    from disk, i.e. from point displacement fields provided by the user in the
    case.

SourceFiles
    meshMotionFromDiskFvMotionSolver.C

Author
    Philip Cardiff, UCD.
    Chanikya Valeti, UCD.

\*---------------------------------------------------------------------------*/

#ifndef meshMotionFromDiskFvMotionSolver_H
#define meshMotionFromDiskFvMotionSolver_H

#include "displacementMotionSolver.H"
#include "fvMotionSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward class declarations
class motionInterpolation;
class motionDiffusivity;

/*---------------------------------------------------------------------------*\
             Class meshMotionFromDiskFvMotionSolver Declaration
\*---------------------------------------------------------------------------*/

class meshMotionFromDiskFvMotionSolver
:
    public displacementMotionSolver,
    public fvMotionSolver
{
    // Private data

        //- Address of the displacement files directory
        const fileName dispFilesDirectory_;


    // Private Member Functions

        //- No copy construct
        meshMotionFromDiskFvMotionSolver
        (
            const meshMotionFromDiskFvMotionSolver&
        ) = delete;

        //- No copy assignment
        void operator=(const meshMotionFromDiskFvMotionSolver&) = delete;


public:

    //- Runtime type information
    TypeName("meshMotionFromDisk");


    // Constructors

        //- Construct from polyMesh and IOdictionary
        meshMotionFromDiskFvMotionSolver
        (
            const polyMesh&,
            const IOdictionary&
        );

        //- Construct from components
        meshMotionFromDiskFvMotionSolver
        (
            const polyMesh& mesh,
            const IOdictionary& dict,
            const pointVectorField& pointDisplacement,
            const pointIOField& points0
        );


    //- Destructor
    ~meshMotionFromDiskFvMotionSolver();


    // Member Functions

        //- Return point location obtained from the current motion field
        virtual tmp<pointField> curPoints() const;

        //- Solve for motion
        virtual void solve();

        //- Update topology
        virtual void updateMesh(const mapPolyMesh&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
