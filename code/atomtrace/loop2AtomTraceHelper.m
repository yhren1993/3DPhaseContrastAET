function [sPeaks] = loop2AtomTraceHelper(vol,sPeaks,numLoops)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions
% 36 - loop over atom tracing components - no peak addition.

tic

minIntensityPeak = 30;
minPeakSigma     = 0.45;

% Main loop
for a0 = 1:numLoops
    % Merge peaks
    sPeaks = mergeAndThresholdAtomTraceHelper(sPeaks);
    % Refine
    sPeaks = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks);
    
    % Delete peaks using extra criteria
    del = ((sPeaks.peaksRefine(:,4) ...
        + sPeaks.peaksRefine(:,6) ...
        < minIntensityPeak)) ...
        | ...
        (sPeaks.peaksRefine(:,5) < minPeakSigma);
    sPeaks.peaksRefine(del,:) = [];
    
    % Remove peaks with regular criteria
    sPeaks = removeSitesAtomTraceHelper(sPeaks);
    if a0 > (numLoops * 0.5)
        % Check local intensity ratio
        sPeaks = removeSitesAtomTraceHelper(sPeaks,true);
    end
    % Refine
    sPeaks = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks);
    
    Np = size(sPeaks.peaksRefine,1);
    disp(['        iter # ' num2str(a0) '/' num2str(numLoops) ... 
        ' done, total number of sites = ' num2str(Np)])
end

toc

end