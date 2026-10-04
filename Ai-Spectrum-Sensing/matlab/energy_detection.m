clc;
clear;
close all;

%% Parameters

Fs = 1000;
T = 0.1;

t = 0:1/Fs:T;

A = 1;
f = 50;

SNR = 10;

%% Generate Primary User Signal

signal = A*sin(2*pi*f*t);

%% Generate Noise

noise_power = 0.5;

noise = sqrt(noise_power) * randn(size(signal));

%% Received Signal

received_signal = signal + noise;

%% Energy Calculation

Energy = sum(received_signal.^2);

%% Noise Energy

NoiseEnergy = sum(noise.^2);

%% Adaptive Threshold

Threshold = 2 * NoiseEnergy;

%% Decision

if Energy > Threshold
    disp("Channel Status : BUSY")
else
    disp("Channel Status : FREE")
end

%% Display Results

fprintf("Signal Energy = %.2f\n", Energy);
fprintf("Noise Energy = %.2f\n", NoiseEnergy);
fprintf("Adaptive Threshold = %.2f\n", Threshold);