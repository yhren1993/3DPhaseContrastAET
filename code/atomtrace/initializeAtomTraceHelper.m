function [sPeaks] = initializeAtomTraceHelper(vol)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions
% 30 - Find initial candidates for peaks from volume

% inputs
sigmaFind = [1 2]/2;  % kernel values for initial peak search
minimumIntensity = 50+50;  % peaks dimmer than this are not included in list
sigmaInit = 0.7; % initial guess for peak standard deviation in pixels 
boundDelete = 50;

sPeaks.settingsInit = [sigmaFind minimumIntensity sigmaInit boundDelete];

% coordinates
sPeaks.volSize = size(vol);
qx = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(1),1),[sPeaks.volSize(1) 1 1]);
qy = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(2),1),[1 sPeaks.volSize(2) 1]);
qz = reshape(makeFourierCoordsAtomTraceHelper(sPeaks.volSize(3),1),[1 1 sPeaks.volSize(3)]);

% correlation volume
% volCorr = fftn(vol);
if sigmaFind(2) == 0
    volCorr = ifftn( ...
        (8^1.5) ...
        *( exp(qx.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qy.^2 * ((-8*pi)*sigmaFind(1)^2)) ...
        .* exp(qz.^2 * ((-8*pi)*sigmaFind(1)^2))) ...
        .* fftn(vol), ...
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
        .* fftn(vol), ...
        'symmetric');
end

% find candidate peak locations
d = -1:1;
[sy,sx,sz] = meshgrid(d,d,d);
dxyz = [sx(:) sy(:) sz(:)];
sub = (dxyz(:,1)==0) & (dxyz(:,2)==0) &  (dxyz(:,3)==0); 
sub = sub | (sum(abs(dxyz),2)==3); 
dxyz(sub,:) = [];
p = volCorr > minimumIntensity;
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
inds = find(p(:));
sizeVol = size(vol);
[xp,yp,zp] = ind2sub(sizeVol,inds);
peaksCand = sortrows([xp yp zp volCorr(p) zeros(numel(xp),1)],-4);

% Update peak positions using parabolic fits
basis = ones(27,1+3);
subFit = true(3,3,3);
subFit(:,:,1) = [0 0 0;0 1 0;0 0 0];
subFit(:,:,2) = [0 1 0;1 1 1;0 1 0];
subFit(:,:,3) = [0 0 0;0 1 0;0 0 0];
subFit = subFit(:);
for a0 = 1:size(peaksCand,1)
    % Crop region around peak
    x = mod(round(peaksCand(a0,1))+(-1:1)-1,sizeVol(1))+1;
    y = mod(round(peaksCand(a0,2))+(-1:1)-1,sizeVol(2))+1;
    z = mod(round(peaksCand(a0,3))+(-1:1)-1,sizeVol(3))+1;
    vc = volCorr(x,y,z);
    
    % Fit parabola functions in 3 dimensions
    dx = (vc(3,2,2) - vc(1,2,2)) ...
        / (4*vc(2,2,2) - 2*vc(3,2,2) - 2*vc(1,2,2));
    dy = (vc(2,3,2) - vc(2,1,2)) ...
        / (4*vc(2,2,2) - 2*vc(2,3,2) - 2*vc(2,1,2));
    dz = (vc(2,2,3) - vc(2,2,1)) ...
        / (4*vc(2,2,2) - 2*vc(2,2,3) - 2*vc(2,2,1));
    
    % Estimate peak intensity
    xc = sx(:) - dx;
    yc = sy(:) - dy;
    zc = sz(:) - dz;
    basis(:,2:end) = [xc.^2 yc.^2 zc.^2];
    beta = basis(subFit,:) \ vc(subFit);
    
    % update peaks - add second intensity estimate
    dx = min(max(dx,-0.5),0.5);
    dy = min(max(dy,-0.5),0.5);
    dz = min(max(dz,-0.5),0.5);
    peaksCand(a0,1:3) = peaksCand(a0,1:3) + [dx dy dz];
    peaksCand(a0,5) = beta(1);
end

% Initialize sPeaks
Np = size(peaksCand,1);
sPeaks.peaksCand = peaksCand;
% peaksRefine - [x y z I0 sigma k_bg]
sPeaks.peaksRefine = double( ...
    [sPeaks.peaksCand(:,[1 2 3 4]) ...
    ones(Np,1)*sigmaInit ones(Np,1)*0]);
sPeaks.peaksID = ones(Np,1);

end