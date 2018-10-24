function [sPeaks] = loop1AtomTraceHelper(vol,sPeaks,numLoops)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions    
% 36 - loop over atom tracing components - with peak addition

tic

minIntensityPeak    = 30;
loopsRefineAfterAdd = 5;

% Main loop
for a0 = 1:numLoops
    % Add new peaks
    sPeaks = addSitesAtomTraceHelper(vol,sPeaks);
    % Refine
    for a1 = 1:loopsRefineAfterAdd
        sPeaks = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks);
    end
    
    % Merge peaks
    sPeaks = mergeAndThresholdAtomTraceHelper(sPeaks);
    % Refine
    sPeaks = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks);
    
    % Delete peaks
    del = sPeaks.peaksRefine(:,4) ...
        + sPeaks.peaksRefine(:,6) ...
        < minIntensityPeak;
    sPeaks.peaksRefine(del,:) = [];
    sPeaks = removeSitesAtomTraceHelper(sPeaks);
    % Refine
    sPeaks = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks);
    
    Np = size(sPeaks.peaksRefine,1);
    disp(['        iter # ' num2str(a0) '/' num2str(numLoops) ... 
        ' done, total number of sites = ' num2str(Np)])
end

toc

end