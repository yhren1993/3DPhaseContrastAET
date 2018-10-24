function [sPeaks,volSubtract] = addSitesAtomTraceHelper(vol,sPeaks)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions    
% Atom tracing - refinement of sites
% 33 - add more atoms

% Inputs
rCut             = 4;
sigmaFind        = [1 2]/2;  % kernel values in pixels  % for 0.5 A pixels, 
minIntensityAdd  = 50+50;
% Density deletion inputs
radiusNN         = 12;   % in pixels / voxels
minNumNNsAllowed = 19;  % if site has less NNs than this, it will be removed
boundDelete      = 50;

sPeaks.settingsAdd = [sigmaFind minIntensityAdd radiusNN minNumNNsAllowed boundDelete];

% coordinates
N = double(size(vol));
[ya,xa,za] = meshgrid(1:N(2),1:N(1),1:N(3));
% rCut = sPeaks.rCut;
v = -rCut:rCut;
Nv = length(v);

% Subtract all peaks from reconstruction volume
peaksRefine = sPeaks.peaksRefine;
volSubtract = vol;
Np = size(peaksRefine,1);
for a0 = 1:Np
    p = peaksRefine(a0,:);
    p(6) = 0;
    
    % Cut out segment around atom
    x = p(1);
    y = p(2);
    z = p(3);
    
    xv = mod(v + round(x) - 1, N(1))+1;
    yv = mod(v + round(y) - 1, N(2))+1;
    zv = mod(v + round(z) - 1, N(3))+1;
    
    xCut = xa(xv,yv,zv);
    yCut = ya(xv,yv,zv);
    zCut = za(xv,yv,zv);
    %     volCrop(:,:,:,a0) = double(vol(xv,yv,zv));
    
    volSubtract(xv,yv,zv) = volSubtract(xv,yv,zv) ...
        - reshape(sPeaks.fun1(p,[xCut(:) yCut(:) zCut(:)]),[1 1 1]*Nv);
end
volSubtract(:) = max(volSubtract,0);


% correlation volume
sPeaks.volSize = size(vol);
qx = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(1),1),[sPeaks.volSize(1) 1 1]);
qy = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(2),1),[1 sPeaks.volSize(2) 1]);
qz = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(3),1),[1 1 sPeaks.volSize(3)]);
if sigmaFind(2) == 0
    volCorr = ifftn( ...
        (8^1.5) ...
        *( exp(qx.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qy.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qz.^2 * ((-8*pi)*sigmaFind(1)^2))) ...
        .* fftn(volSubtract), ...
        'symmetric');
else
    volCorr = ifftn( ...
        (8^1.5) * ( ...
           exp(qx.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qy.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qz.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        -  exp(qx.^2 * ((-8*pi)*sigmaFind(2)^2)) ...
        .* exp(qy.^2 * ((-8*pi)*sigmaFind(2)^2)) ...
        .* exp(qz.^2 * ((-8*pi)*sigmaFind(2)^2)) ) ...
        .* fftn(volSubtract), ...
        'symmetric');
end

% find candidate peak locations
d = -1:1;
[sy,sx,sz] = meshgrid(d,d,d);
dxyz = [sx(:) sy(:) sz(:)];
sub = (dxyz(:,1)==0) & (dxyz(:,2)==0) &  (dxyz(:,3)==0); 
sub = sub | (sum(abs(dxyz),2)==3); 
dxyz(sub,:) = [];
p = volCorr > minIntensityAdd;
for a0 = 1:size(dxyz,1)
    p(:) = p & ...
        volCorr > circshift(volCorr,dxyz(a0,:));
end
p(1:boundDelete,:,:) = false;
p(:,1:boundDelete,:) = false;
p(:,:,1:boundDelete) = false;
p(((1-boundDelete):0)+size(p,1),:,:) = false;
p(:,((1-boundDelete):0)+size(p,2),:) = false;
p(:,:,((1-boundDelete):0)+size(p,3)) = false;


% Convert to [x y z intensity], sort from brightest to dimmest
sizeVol = size(vol);
inds = find(p(:));
[xp,yp,zp] = ind2sub(sizeVol,inds);
peaksCand = sortrows([xp yp zp volSubtract(p) zeros(numel(xp),1)],-4);
% use density criteria to remove peaks


% Refine new peaks from NN subtracted volume
Np = size(peaksCand,1);
sPeaksNew.peaksRefine = zeros(Np,6);
sPeaksNew.peaksRefine(:,1:3) = peaksCand(:,1:3);
sPeaksNew.peaksRefine(:,4:6) = ...
    repmat(mean(sPeaks.peaksRefine(:,4:6),1),[Np 1]);

% Delete peaks below density threshold
peaksRefineUnion = [sPeaksNew.peaksRefine; sPeaks.peaksRefine];
del = false(size(peaksRefineUnion,1),1);
r2max = radiusNN^2;
for a0 = 1:size(peaksRefineUnion,1)
    d2 =  (peaksRefineUnion(a0,1) - peaksRefineUnion(:,1)).^2 ...
        + (peaksRefineUnion(a0,2) - peaksRefineUnion(:,2)).^2 ...
        + (peaksRefineUnion(a0,3) - peaksRefineUnion(:,3)).^2;
    numNN = sum(d2<r2max) - 1;
    if numNN < minNumNNsAllowed
        del(a0) = true;
    end
end
delSub = del(1:size(sPeaksNew.peaksRefine,1));
sPeaksNew.peaksRefine(delSub,:) = [];

% Output number of peaks added
numPeaksNew = size(sPeaksNew.peaksRefine,1);
disp([num2str(numPeaksNew) ' peaks added, refining locations ...'])

% Refine peak locations from NN subtracted volume
sPeaksNew = refineSitesAtomTraceHelper(volSubtract,sPeaksNew);

sPeaks.peaksRefine = [ ...
    sPeaks.peaksRefine; sPeaksNew.peaksRefine];

end