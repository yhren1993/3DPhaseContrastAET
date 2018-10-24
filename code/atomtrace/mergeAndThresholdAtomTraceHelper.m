function [sPeaks] = mergeAndThresholdAtomTraceHelper(sPeaks)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions
% 34 - Merge atoms that are too close together, or below intensity threshold
% If no output is specified, graph pair distribution function (PDF).
distMerge = 2.25;  % in voxels
voxelSize = 0.5;  % In Angstroms, just for plotting


sPeaks.settingsMerge = [distMerge voxelSize];

if nargout == 0
    dr = 0.05;
    rRange = [0 20];
    r = (rRange(1):dr:rRange(2))';
    Nr = length(r);
    
    skip = 37;  % speed up calc
    %     skip = 7;  % speed up calc
    
    
    Np = size(sPeaks.peaksRefine,1);
    counts = zeros(Nr,1);
    r2max = (rRange(2) - 1e-8)^2;
    for a0 = 1:skip:Np
        d2 =  (sPeaks.peaksRefine(a0,1) - sPeaks.peaksRefine(:,1)).^2 ...
            + (sPeaks.peaksRefine(a0,2) - sPeaks.peaksRefine(:,2)).^2 ...
            + (sPeaks.peaksRefine(a0,3) - sPeaks.peaksRefine(:,3)).^2;
        sub = d2<r2max;
        sub(a0) = false;
        rInds = round((sqrt(d2(sub)) - rRange(1)) / dr) + 1;
        
        if ~isempty(rInds)
            counts(:) = counts + accumarray(rInds,ones(length(rInds),1),[Nr 1]);
        end
    end
    
    figure(32)
    clf
    %     plot(r,counts*voxelSize,'linewidth',2,'color','r')
        plot(r(2:end)*voxelSize,counts(2:end) ./ r(2:end).^2,...
            'linewidth',2,'color','r')
    xlabel('Distance [A]')
    
else
    
    % Merge peaks too close together
    Np = size(sPeaks.peaksRefine,1);
    
    % set up NN network
    volInds = zeros(sPeaks.volSize);
    inds = sub2ind(sPeaks.volSize,...
        round(sPeaks.peaksRefine(:,1)),...
        round(sPeaks.peaksRefine(:,2)),...
        round(sPeaks.peaksRefine(:,3)));
    volInds(inds) = 1:Np;
    
    % init
    mark = false(Np,1);
    del = false(Np,1);
    peaksRefine = sPeaks.peaksRefine;
    
    % loop over sites
    r2max = distMerge^2;
    vCut = (floor(-distMerge-1):ceil(distMerge+1))-1;
    for a0 = 1:Np
        if mark(a0) == false
            volCrop = volInds( ...
                mod(round(sPeaks.peaksRefine(a0,1))+vCut,sPeaks.volSize(1))+1,...
                mod(round(sPeaks.peaksRefine(a0,2))+vCut,sPeaks.volSize(2))+1,...
                mod(round(sPeaks.peaksRefine(a0,3))+vCut,sPeaks.volSize(3))+1);
            indsCheck = volCrop(volCrop>0);
            
            d2 =  (sPeaks.peaksRefine(a0,1) - sPeaks.peaksRefine(indsCheck,1)).^2 ...
                + (sPeaks.peaksRefine(a0,2) - sPeaks.peaksRefine(indsCheck,2)).^2 ...
                + (sPeaks.peaksRefine(a0,3) - sPeaks.peaksRefine(indsCheck,3)).^2;
            subCheck = d2<r2max;
            indsNN = indsCheck(subCheck);
            indsNN(mark(indsNN) == true) = [];
            
            Nm = length(indsNN);
            if Nm > 1
                % Merge event
                %mark(sub) = true;
                %inds = find(sub);
                mark(indsNN) = true;
                
                % Weight by intensity * sigma
                pNew = zeros(1,6);
                weightTotal = 0;
                for a1 = 1:length(indsNN)
                    w = peaksRefine(indsNN(a1),4) ...
                        * peaksRefine(indsNN(a1),5);
                    pNew = pNew + w*peaksRefine(indsNN(a1),:);
                    weightTotal = weightTotal + w;
                end
                pNew = pNew / weightTotal;
                peaksRefine(indsNN(1),:) = pNew;
                
                % peaksRefine(indsNN(1),:) = mean(peaksRefine(indsNN,:),1);
                del(indsNN(2:end)) = true;
                % scale intensity?
                peaksRefine(indsNN(1),4) = peaksRefine(indsNN(1),4) * Nm;
            end
        end
    end
    
    disp([num2str(sum(del)) ' peaks removed by merging'])
    peaksRefine(del,:) = [];
    sPeaks.peaksRefine = peaksRefine;
end

end