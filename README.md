# ScanEP


ScanEP is designed to calculate the parameters of hopping transport according to Marcus theory in organic amorphous 
and crystalline semiconductors using the FireFly software package. The calculations are performed using the CASSCF and 
XMCQDPT methods. It includes the following modules:
  - Molecular delimiter
  - Inp files generator
  - Method changer
  - VEC Changer
  - QDPT result processing

### Molecular delimiter
> If you have molecular modeling cell or crystal fragment (typically 2×2×2 to 3×3×3 unit cells) and you need monomers 
> and dimers you can use this module
  - "Path to initial file" is the path to file with atomic coordinates in xyz format
  - "Path to output folder" is the path to directory where new files will be generated
  - "Minimum atoms per molecule" is the minimum size of the monomer (in order to remove molecular fragments that can be 
present as a result of incorrect extraction of the crystal fragment)
  - "Molecular bond limit" is the maximum bond length in the monomer
  - Select "Merge monomers to dimers" if you need to generate dimers
  - "Maximum contact length" is the maximum contact length between monomers to avoid generating too many dimers at 
large distance 

### Molecular interpolator

> Interpolator interpolates molecular structure in internal coordinates between two states given in Cartesian 
> coordinates. Initial and final states are built manually. More information you can find in []. The resulting files 
> are in GAMESS input format with some default header and coordinates in z-matrix format
  - "Path to initial point" is the path to file with initial geometry in xyz format
  - "Path to final point" is the path to file with final geometry in xyz format 
  - "Path to output folder" is the path to directory where new files will be generated
  - "Amount of steps" is the desired number of points on the energy profile including the initial and final ones
  - Select "Coordinates to XYZ" to print cartesian atomic coordinates instead of z-matrix

### Method changer
> Method changer replaces the default input file  header with a user-specified one in all files within the folder.
  - "Path to file or folder" is the path to folder with .inp files to change
  - "Path to new method file" is the path to the new header file
  - "Path to output folder" is the path to directory where new files will be placed

### VEC Changer
>Change $VEC group adds or replaces the $VEC group in all files within the folder.
  - "Path to file or folder" is the path to folder with .inp files to change
  - "Path to new $VEC file" is the path to the new $VEC file (from converged RHF or CASSCF calculation)
  - "Path to output folder" is the path to directory where new files will be placed

### QDPT result processing
>To process the results of XMCQDPT calculation, press "Plot editor" button to go to the plot editor menu. To return to 
> the main menu close plot editor window. The program also calculates Reorganization energy (Ereorg), Site energy 
> disorder DeltaG, hopping integral (Energy gap) and hopping rate constant (KHop) at the given temperature
  - "Path to  QDPT files" is the path to the folder with XMCQDPT .out files
  - Other fields are self-explanatory

## Installation

### Running
Download all files in directory .../molecule_processor/

Install python and libraries from requirements.txt
> pip install -r requirements.txt

Run the program
> python molecule_processor.py

### Build to exe
Download all files in directory .../molecule_processor/

Install python and libraries from requirements.txt
> pip install -r requirements.txt

Install pyinstaller
> pip install pyinstaller
