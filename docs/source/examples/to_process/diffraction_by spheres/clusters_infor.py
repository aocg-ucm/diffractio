% CALCULO_AJUSTES_COMSOL Calcula los ajustes de scattering de clusters y ves�culas
% usando los datos de Comsol
%
% @ 2018 Infor, AOCG

clear all; clc; close all;

% Par�metros generales
r = linspace(20,200,20); % Radio del cluster (nm)
c = linspace(0.1,0.5,20); % Concentracion volumetrica relativa

% 1. Calculo de vesiculas (solo dependen de la concentraci�n)
% 1.1. Scattering cross section vesiculas
% Par�metros
Sv1 = 1.89802e-12;
Sv2 = -7.44055e-12;
Sv3 = 5.77093e-12;
Sv = Sv1*c.^2+Sv2*c+Sv3;
plot(c,Sv);
title('Scattering cross section (Vesiculas)');
xlabel('Concentraci�n (c)');
ylabel('\sigma (m^2)');

% 1.2 Coeficiente asimetria vesiculas
% Parametros
gv1 = 0.109884;
gv2 = 0.041627;
gv3 = 0.770523;
g = gv1*c.^2+gv2*c+gv3;
figure;
plot(c,g);
title('Coef. asimetria (Vesiculas)');
xlabel('Concentraci�n (c)');
ylabel('g');

% 2 Calculo para clusters (dependen del radio y concentracion)
[R,C] = meshgrid(r,c);
% 2.1 Calculo del scattering cross-section (vesiculas)
Sp00 = 1.24e-15;
Sp10 = 2.577e-15;
Sp01 = -5.059e-16;
Sp20 = 2.153e-15;
Sp11 = -1.162e-15;
Sp02 = 5.071e-17;
Sp30 = 8.025e-16;
Sp21 = -1.055e-15;
Sp12 = 1.834e-16;
Sp40 = 9.334e-17;
Sp31 = -3.366e-16;
Sp22 = 1.102e-16;

% Parametros normalizados (necesarios para Cross Section)
rcm = 112.5;
rcstd = 51.95;
Rn = (R-rcm)/rcstd;
clocm = 0.35;
clocstd = 0.08805;
Cn = (C-clocm)/clocstd;

% Calculo del scattering  cross section (como lo hago en la dll)
Stemp = Sp40;
Stemp = Stemp.*Rn + Sp31.*Cn + Sp30;
Stemp = Stemp.*Rn + Sp22.*Cn.^2 + Sp21.*Cn + Sp20;
Stemp = Stemp.*Rn + Sp12.*Cn.^2 + Sp11.*Cn + Sp10;
Stemp = Stemp.*Rn + Sp02.*Cn.^2 + Sp01.*Cn + Sp00;
Sc = Stemp;

% Representacion resultados
figure;
plot(r,Sc(1:5:end,:));
title('Scattering cross section (Cluster)');
xlabel('Radio cluster (nm)');
ylabel('\sigma (m^2)');
cv = c(1:5:end);
text = cell(1,length(cv));
for k=1:length(cv)
    text{k} = ['c = ' num2str(cv(k),3)];
end
legend(text);

% 2.2 Calcula el coeficiente  de asimetria (clusters)
% Parameters for computing the anisotropy coefficient (fromn Alex's Comsol simulations)*/
gp00 = -0.01276;
gp10 = -0.001197;
gp01 = -0.04586;
gp20 = 8.628e-5;
gp11 = 0.00237;
gp02 = 0.01374;
gp30 = -3.046e-7;
gp21 = -9.82e-6;
gp12 = -0.0002186;

% Computing the anysotropy coefficient*/
gtemp = gp30;
gtemp = gtemp.*R + gp20 + gp21.*C;
gtemp = gtemp.*R + gp10 + gp11.*C + gp12.*(C.^2);
gtemp = gtemp.*R + gp00 + gp01.*C + gp02.*(C.^2);
gc = gtemp;

% Representaci�n gr�fica
figure;
plot(r,gc(1,:),r,gc(end,:));
title('Coeficiente asimetria (Clusters)');
xlabel('Radio cluster (nm)');
ylabel('g');
cv = [c(1) c(end)];
text = cell(1,length(cv));
for k=1:length(cv)
    text{k} = ['c = ' num2str(cv(k),3)];
end
legend(text);
