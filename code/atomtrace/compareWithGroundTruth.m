function [sPeaks] = compareWithGroundTruth(sPeaks, atoms_ground_truth)
tic
% compare traced atomic coordinates with input coords (ground truth)

radiusMax = 2;    % in voxels
xCutoff   = 180;  % keep only tip diameter <12 nm
voxelSize = 0.5;  % voxel size in Angstroms

% mark atoms as they are paired, to prevent double counting
N = size(atoms_ground_truth,1);
xyzID = [atoms_ground_truth(:,1:4) (1:N)'];

% comparison array:
% [x y z ID index_ground_truth]
Np = size(sPeaks.peaksRefine,1);
sPeaks.peaksCompareGT = zeros(Np,5);

% main loop
r2 = radiusMax^2;
subBoundary = sPeaks.peaksRefine(:,1) > xCutoff;
[~,indsOrder] = sort(sPeaks.peaksRefine(:,4),'descend');
for a0 = 1:Np
    indCheck = indsOrder(a0);
    
    if size(xyzID,1) > 0
        [val,ind] = min( ...
              (xyzID(:,1) - sPeaks.peaksRefine(indCheck,1)).^2 ...
            + (xyzID(:,2) - sPeaks.peaksRefine(indCheck,2)).^2 ...
            + (xyzID(:,3) - sPeaks.peaksRefine(indCheck,3)).^2);
        if val < r2
            sPeaks.peaksCompareGT(indCheck,1:5) = xyzID(ind,:);
            xyzID(ind,:) = [];
            subBoundary(indCheck) = sPeaks.peaksCompareGT(indCheck,1) > xCutoff;
        end
    end
end
toc

% Output other pieces
sPeaks.atoms_ground_truth = atoms_ground_truth;
sPeaks.atomsResidual = xyzID;

% Measurements
sub = (sPeaks.peaksCompareGT(:,5) > 0) & (sPeaks.peaksCompareGT(:,1) > xCutoff);
sPeaks.indicesFound = sub;
dxyz = sPeaks.peaksRefine(sub,1:3) - sPeaks.peaksCompareGT(sub,1:3);
distError = sqrt(sum(dxyz.^2,2));
sPeaks.peaksPositionError = [dxyz distError];
sPeaks.numPeaks = sum(subBoundary);
sPeaks.numPeaksFound = sum(sub);
sPeaks.numPeaksFalsePositive = sPeaks.numPeaks - sPeaks.numPeaksFound;

sub = sPeaks.atomsResidual(:,1) > xCutoff;
sPeaks.numPeaksMissing = sum(sub);
sPeaks.numPeaksTotal = sPeaks.numPeaksFound + sPeaks.numPeaksMissing;

% Output some results
disp([num2str(sPeaks.numPeaks) ' peaks found, ' ...
    num2str(sPeaks.numPeaksMissing) ' missing peaks, ' ...
    num2str(sPeaks.numPeaksFalsePositive) ' false positives.' ...
    ])
disp([sprintf('%.04f',100* ...
    (sPeaks.numPeaksMissing/sPeaks.numPeaksTotal)) ...
    ' % of peaks missing, ' ...
    sprintf('%.04f',100*sPeaks.numPeaksFalsePositive/sPeaks.numPeaksTotal) ...
    ' % of peaks determined to be false positives.' ...
    ])

% Plot RMS error histogram
figure(567)
clf
hist(sPeaks.peaksPositionError(:,4)*voxelSize*100,128)
xlabel('Position Error [pm]')


end