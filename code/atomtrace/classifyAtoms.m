function sPeaks = classifyAtoms(sPeaks)
% David Ren - April 2018 - atom classification for phase contrast reconstructions    
%This function classifies atoms and computes error rate.
silicon_mass = 28;
oxygen_mass  = 16;
atom_diff    = silicon_mass - oxygen_mass;

voxelSize = 0.5;  % in Angstroms
powerSigma = 1;

xHist = (0:2:100);
sig = sPeaks.peaksRefine(:,4) ...
    .* ((sPeaks.peaksRefine(:,5) * voxelSize).^powerSigma);
yHist = hist(sig,xHist);

% Fit intensity of atoms by Gaussians 
fun1 = @(c,x) ...
      c(1) * exp((-1/(2*c(3)^2)) *((x-c(5)).^2)) ...
    + c(2) * exp((-1/(2*c(4)^2)) *((x-c(6)).^2));
options = optimset(...
    'TolX',1e-4,...
    'TolFun',1e-4,...
    'MaxFunEvals',10^4,...
    'display','off');
coeff = [4000 4000 10 10 35 70];
coeff = lsqcurvefit(fun1,coeff,xHist,yHist,[],[],options);
c = (coeff(4)^2*coeff(5)^2 - coeff(3)^2*coeff(6)^2) - 2 * coeff(3)^2*coeff(4)^2*(log(coeff(1)/coeff(2)));
b = -2 * (coeff(4)^2*coeff(5)-coeff(3)^2*coeff(6));
a = (coeff(4)^2-coeff(3)^2);

x_intersect = (-b - sqrt(b^2 - 4 * a * c)) / (2 * a);
if ((x_intersect - coeff(5)) * (x_intersect - coeff(6))) > 0
    x_intersect = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);
    fprintf('x_intersect: %.04f\n',x_intersect);
end

% Classify according to intersection of gaussian
sPeaks.histIntersect = x_intersect;
atoms_species = oxygen_mass + atom_diff * (sig(sPeaks.indicesFound) > x_intersect);
true_species = sPeaks.peaksCompareGT(sPeaks.indicesFound,4);

% Count total number of correct species
sPeaks.numCorrectSpecies = sum(abs(true_species - atoms_species) < 1e-8);

disp([sprintf('%.04f',100* ...
    (sPeaks.numCorrectSpecies/sPeaks.numPeaksFound)) ...
    ' % of peaks correctly identified' ])

end

