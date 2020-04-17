function [sPeaks] = refineSitesAtomTraceHelper(vol,sPeaks)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions    
% Atom tracing - refinement of sites
% 31 - nonlinear least squares fitting

% STEM fits
NsubPxIter  = 4;         % Outer loop fitting iterations.
rCut        = 3;               % Cutting radius (to speed up fits).
rFit        = 1.8;               % fitting radius in pixels.
xyzStep     = 0.25;           % maximum change in x/y per iteration.
f_plot      = 1*0;             % To plot as you go, set to "true" or 1 (for tuning)
sigmaRange  = [0.4 1.2];  % Max and min sigma values IN PIXELS
sigmaBounds = [0.75 1.25];    % min and max allowed sigma values relative to init
intBounds   = [0.5 1.5];    % min and max allowed int values relative to init
bgRelative  = 0.25;  % How much background can change w.r.t. intensity
bgRange     = [0 1e-6];  % background range

fun1 = @(c,x) c(6) + c(4)* ...
    exp((-1/(2*c(5)^2)) ...
    *((x(:,1)-c(1)).^2 + (x(:,2)-c(2)).^2 + (x(:,3)-c(3)).^2));
% Scale tolerance to reconstruction values
signalValue = mean(vol(vol>max(vol(:))*0.5));
options = optimset(...
    'TolX',1e-2 * signalValue,...
    'TolFun',1e-2 * signalValue,...
    'MaxFunEvals',10^3,...
    'display','off');

% Output function and options
sPeaks.fun1 = fun1;
sPeaks.options = options;
sPeaks.rCut = rCut;

% coordinates
N = double(size(vol));
[ya,xa,za] = meshgrid(1:N(2),1:N(1),1:N(3));
v = -rCut:rCut;
r2 = rFit^2;

% Extract peaks to try parfor
Np = size(sPeaks.peaksRefine,1);
peaksRefine = sPeaks.peaksRefine;

% Pre-calc cropped regions
Nv = length(v);
Ncrop = [Nv Nv Nv Np];
xCrop = zeros(Ncrop);
yCrop = zeros(Ncrop);
zCrop = zeros(Ncrop);
volCrop = zeros(Ncrop);
for a0 = 1:Np
    p = peaksRefine(a0,:);
    
    % Cut out segment around atom
    x = p(1);
    y = p(2);
    z = p(3);
    
    xv = mod(v + round(x) - 1, N(1))+1;
    yv = mod(v + round(y) - 1, N(2))+1;
    zv = mod(v + round(z) - 1, N(3))+1;
    
    xCrop(:,:,:,a0) = xa(xv,yv,zv);
    yCrop(:,:,:,a0) = ya(xv,yv,zv);
    zCrop(:,:,:,a0) = za(xv,yv,zv);
    volCrop(:,:,:,a0) = double(vol(xv,yv,zv));

end


% main fitting loop
% for a0 = 1:1111:Np
parfor a0 = 1:Np
    p = peaksRefine(a0,:);
    p(6) = min(max(p(6),bgRange(1)),bgRange(2));
    
    x = p(1);
    y = p(2);
    z = p(3);
    
    xCut = xCrop(:,:,:,a0);
    yCut = yCrop(:,:,:,a0);
    zCut = zCrop(:,:,:,a0);
    volCut = volCrop(:,:,:,a0);
    
    % Inner loop
    for a1 = 1:NsubPxIter
        % Subset
        sub = (xCut-x).^2 + (yCut-y).^2 + (zCut-z).^2 < r2;
        if sum(sub) > 6
            
            xfit = xCut(sub);
            yfit = yCut(sub);
            zfit = zCut(sub);
            volfit = volCut(sub);
            
            % Upper and lower bounds
            bgChange = bgRelative * p(4);
            lb = [p(1:3)-xyzStep p(4)*intBounds(1) ...
                p(5)*sigmaBounds(1) p(6)-bgChange];
            ub = [p(1:3)+xyzStep p(4)*intBounds(2) ...
                p(5)*sigmaBounds(2) p(6)+bgChange];
            lb(5) = max(lb(5),sigmaRange(1));
            ub(5) = min(ub(5),sigmaRange(2));
            lb(6) = max(lb(6),bgRange(1));
            ub(6) = min(ub(6),bgRange(2));
            lb([4 6]) = max(lb([4 6]),0);
            
            % Perform fitting
            p = lsqcurvefit(fun1,p,[xfit yfit zfit],volfit,lb,ub,options);
            
            % Plotting if needed
            if f_plot == 1
                volTest = reshape(fun1(p, ...
                    [xCut(:) yCut(:) zCut(:)]),size(volCut));
                Ip1 = [squeeze(volCut(rCut+1,:,:)) ...
                    squeeze(volCut(:,rCut+1,:)) ...
                    volCut(:,:,rCut+1)];
                Ip2 = [squeeze(volTest(rCut+1,:,:)) ...
                    squeeze(volTest(:,rCut+1,:)) ...
                    volTest(:,:,rCut+1)];
                
                figure(1)
                clf
                imagesc([Ip1; Ip2])
                axis equal off
                colormap(hot(256))
                caxis([min(Ip2(:)) max(Ip2(:))])
                drawnow
            end
        else
            peaksRefine(a0,:) = p;
        end
    end
    
    % Output result of fit
    peaksRefine(a0,:) = p;
end
sPeaks.peaksRefine = peaksRefine;

end