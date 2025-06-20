%{probability density vs position}
clc; clear; close all;
% Constants
hbar = 1.0545718e-34;
m = 9.10938356e-31;
e = 1.602176634e-19;
eps0 = 8.854187817e-12;
A = e^2 / (4 * pi * eps0);
% Domain
x = linspace(-5e-10, 5e-10, 1000); % Position in meters
n_values = 1:5;

figure;
hold on;
title('Probability Density |\psi_n(x)|^2 vs. x'); % Corrected quotes
xlabel('x (m)'); % Corrected quotes
ylabel('|\psi_n(x)|^2'); % Corrected quotes

for n = n_values
    kappa = m * A / (hbar^2 * n);
    % Corrected psi function: Removed sign(x) for mathematical consistency.
    % This will make all wavefunctions (and thus probability densities) even.
    psi = sqrt(kappa) * 2 .* kappa .* abs(x) .* exp(-kappa .* abs(x));

    % Normalize psi before calculating probability density
    % This is crucial for correct probability interpretation.
    norm_factor = sqrt(trapz(x, abs(psi).^2));
    psi = psi / norm_factor;

    prob_density = abs(psi).^2;
    plot(x, prob_density, 'DisplayName', ['n = ' num2str(n)]); % Corrected quotes
end

legend('Location', 'best'); % Added legend to display labels
grid on;
hold off;