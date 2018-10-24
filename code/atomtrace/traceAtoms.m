function [sPeaks] = traceAtoms(vol)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions	
% Overall schedule control script for atomic refinement

num_iteration1 = 1;
num_iteration2 = 11;
sPeaks = initializeAtomTraceHelper(vol);
sPeaks = refineSitesAtomTraceHelper(vol,sPeaks);

sPeaks = loop1AtomTraceHelper(vol,sPeaks,num_iteration1);
sPeaks = loop2AtomTraceHelper(vol,sPeaks,num_iteration2);

end