%{Energy vs wave number}
clc; clear; close all;
% Constants
hbar = 1.0545718e-34;
m = 9.10938356e-31;
e = 1.602176634e-19;
eps0 = 8.854187817e-12;
A = e^2 / (4 * pi * eps0);
% Quantum numbers
n_values = 1:5;
% Compute wave numbers and energies
kappa = m * A ./ (hbar^2 * n_values);
energy = -m * A^2 ./ (2 * hbar^2 * n_values.^2);
% Plot energy vs wave number
figure;
plot(kappa, energy, 'o-', 'LineWidth', 2); % Corrected quotes
title('Energy vs. Wave Number \kappa_n'); % Corrected quotes
xlabel('\kappa_n (1/m)'); % Corrected quotes
ylabel('Energy E_n (J)'); % Corrected quotes
grid on;

%{Energy vs Quantum}
clc; clear; close all;
% Constants
hbar = 1.0545718e-34;
m = 9.10938356e-31;
e = 1.602176634e-19;
eps0 = 8.854187817e-12;
A = e^2 / (4 * pi * eps0);
% Quantum numbers
n_values = 1:5;
% Compute energies
energy = -m * A^2 ./ (2 * hbar^2 * n_values.^2);
% Plot energy vs quantum number
figure;
plot(n_values, energy, 's-', 'LineWidth', 2); % Corrected quotes
title('Energy E_n vs. Quantum Number n'); % Corrected quotes
xlabel('n'); % Corrected quotes
ylabel('Energy E_n (J)'); % Corrected quotes
grid on;