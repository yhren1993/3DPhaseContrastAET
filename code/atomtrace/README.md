# Atom Tracing and Classifying code
**Written in MATLAB by Colin Ophus and David Ren**

The functions in this directory are used to trace atoms in a final reconstruction volume and compare with with ground truth.
The main functions in the directory are:
1. ```[sPeaks] = traceAtoms(vol)```
	- ```vol```: the 3D reconstructed volume (3D matlab array)
	- ```sPeaks```: tracing result (matlab structure)

2. ```[sPeaks] = compareWithGroundTruth(sPeaks, atomsGroundTruth)```
	- ```sPeask```: tracing result (matlab structure)
	- *```atoms_ground_truth```: ground truth all N atom locations (2D matlab array of size N * 4, x,y,z,and z-number of each and every atom).

3. ```sPeaks = classifyAtoms(sPeaks)```
	- ```sPeaks```: tracing results (matlab structure) **After passing out of ```compareWithGroundTruth``` in 2**

*```atoms_ground_truth```: accessible under data/atomtrace/.
