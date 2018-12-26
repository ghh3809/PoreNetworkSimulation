clear; clc;

load('initial_pressure_2_1.mat');

plot(linspace(0, 1, 100), linear_pressure, 'k', 'LineWidth', 2);
hold on;
plot(linspace(0, 1, 100), gas_pressure, 'k--', 'LineWidth', 2);
plot(linspace(0, 1, 100), scale_pressure, 'k:', 'LineWidth', 2);
title('Initial Pressure (P_{in} = 0.2MPa, P_{out} = 0.1MPa)');
xlabel('Position');
ylabel('Pressure');
legend('Linear (Liquid)', 'Quadratic (Gas)', 'Scale Effect (Gas)');