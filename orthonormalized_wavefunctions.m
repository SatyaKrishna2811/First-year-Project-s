clc; clear all; close all;
% === Constants ===
hbar = 1.0545718e-34; % Reduced Planck’s constant (J·s)
m = 9.10938356e-31; % Electron mass (kg)
e = 1.602176634e-19; % Elementary charge (C)
eps0 = 8.854187817e-12; % Vacuum permittivity (F/m)
A = e^2 / (4 * pi * eps0); % Coulomb potential constant
% === Domain and Setup ===
x_nm = linspace(-5, 5, 1000); % Position in nanometers
x = x_nm * 1e-9; % Convert x to meters
dx = x(2) - x(1);
n_values = 1:5;
N = length(n_values);
psi_raw = zeros(N, length(x)); % Raw (possibly non-orthogonal) wavefunctions
% === Construct & Normalize Raw Wavefunctions ===
for i = 1:N
n = n_values(i);
kappa = m * A / (hbar^2 * n);
% Corrected psi function: removed sign(x) and adjusted for a common exponential decay form.
% Note: This is NOT the general 1D Hydrogen atom wavefunction for n > 1.
% It represents a simple exponentially decaying function, leading to even wavefunctions.
psi = sqrt(kappa) * (2 * kappa * abs(x)) .* exp(-kappa * abs(x));
psi = psi / sqrt(trapz(x, abs(psi).^2)); % Normalize
psi_raw(i, :) = psi;
end
% === Gram-Schmidt Orthonormalization ===
psi_ortho = zeros(N, length(x));
for i = 1:N
psi_ortho(i, :) = psi_raw(i, :);
for j = 1:i-1
proj = trapz(x, conj(psi_ortho(j, :)) .* psi_ortho(i, :));
psi_ortho(i, :) = psi_ortho(i, :) - proj * psi_ortho(j, :);
end
% Normalize the orthonormalized wavefunction
norm_factor = sqrt(trapz(x, abs(psi_ortho(i, :)).^2));
psi_ortho(i, :) = psi_ortho(i, :) / norm_factor;
end
% === Check Orthonormality Matrix ===
S = zeros(N);
for i = 1:N
for j = 1:N
S(i,j) = trapz(x, conj(psi_ortho(i,:)) .* psi_ortho(j,:));
end
end
% === Display Overlap Matrix ===
disp('Orthonormalized Overlap Matrix S = <_i|_j>:'); % Corrected disp syntax
disp(abs(S)); % Take absolute to check closeness to 0 or 1
% === Plot Orthonormalized Wavefunctions ===
figure;
hold on;
title('Orthonormalized Wavefunctions \psi_n(x) vs. x');
xlabel('x (nm)');
ylabel('\psi_n(x)');
for i = 1:N
plot(x_nm, psi_ortho(i,:), 'DisplayName', ['n = ' num2str(n_values(i))]);
end
legend();
grid on;
hold off;