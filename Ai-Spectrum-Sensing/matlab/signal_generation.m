clc;
clear;
close all;

%% Signal Parameters

Fs = 1000;          % Sampling Frequency (Hz)           
t = 0:1/Fs:0.1;

A = 1;              % Amplitude
f = 50;             % Signal Frequency (Hz)

%% Generate Signal

signal = A*sin(2*pi*f*t);

%% Plot

subplot(2,1,1)
plot(t,signal,'r','LineWidth',2);
grid on;
title('Generated Wireless Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');

% Add noise to the original signal

SNR = 10;
received_signal = awgn(signal,SNR,'measured');

subplot(2,1,2)
plot(t,received_signal,'b','LineWidth',2);
grid on;
title('Received Signal with AWGN');
xlabel('Time');
ylabel('Amplitude');

% calculate energy

Energy = sum(received_signal.^2);

disp(Energy)