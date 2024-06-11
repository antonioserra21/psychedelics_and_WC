clear;
clc;
clf;
tic;

%Number of neurons in the simulation
NE          = 50;    
NI          = 50;    
N_neur      = NE+NI;

%Connection strength 
wEE         = 16/10;        %Strength of self-excitatory feedback
wEI         = -26/10;       %Strength of connections between E–I
wIE         = 2;            %Strength of connections between I-E
wII         = -1;           %Strength of self-inhibitory feedback

%External input
In_ext_E     = 2;
In_ext_I_rng = 20:-1:1;     %Range of decreasing in current to inh neurons
                            %to simulate disinhibition

%Simulation parameters
T           = 3000;               %Simulation duration in ms
dt          = 1;              %Time step
data_points = length(0:dt:T);
fs = 1/(dt*1e-3);       % sampling frequency
freqs       = 1:200;

%Number of trials
data_trials = 10;
                                    
%Creting connectivity matrix
W           = zeros(N_neur);
W(1:NE,1:NE) = wEE;
W(1:NE,1+NE:end) = wEI;
W(1+NE:end,1:NE) = wIE;
W(1+NE:end,1+NE:end) = wII;
W = W.*(1+.1*rand(size(W)));    %Adding some randomness to the connectivity matrix
W(eye(length(W))==1)=0;         %No recurrent cennections

data_r = zeros(NE+NI, length(0:dt:T), data_trials);

%Arrays containing info about peak frequency at different levels of inhibiton for each trial
peakFreqE = zeros(data_trials, length(In_ext_I_rng)); 
peakFreqI = zeros(data_trials, length(In_ext_I_rng));

%Arrays containing info about the integral of frequency around its peak
integral_peak_E = zeros(data_trials, length(In_ext_I_rng));
integral_peak_I = zeros(data_trials, length(In_ext_I_rng));
% Define the bandwidth for integration around the peak frequency
bandwidth = 14;  % For example, ±7 Hz around the peak

for k = 1:data_trials %Rep for each trial
    k
    indx = 1;
    for In_ext_I = In_ext_I_rng % Rep for each inhibitory current value

        r = simulate_WC(W, T, dt, In_ext_E, In_ext_I, NE, NI); % Wilson-Cowan results
        data_r(:, :, k) = r;
        rE = r(1:NE,:);
        rI = r(1+NE:end,:);

        mean_rE = mean(rE, 1); 
        mean_rI = mean(rI, 1);

        % Compute the Fast Fourier Transform (FFT) of rE and rI
        
        t_trans = 200;
        id_trans = t_trans/dt;

        X = mean_rE(id_trans:end);
        Y = mean_rI(id_trans:end);
        
        X = X - nanmean(X);
        Y = Y - nanmean(Y);

        FFT_rE = fft(X);
        FFT_rI = fft(Y);
        
        N = length(FFT_rE);
        
        % Set parameters for the Spectral Analysis
        
        f = (0:N-1)*(fs/N);     % frequency range
        power_rE = abs(FFT_rE).^2/N;    % power of the DFT for E population
        power_rI = abs(FFT_rI).^2/N;
        
        [maxValueE, indexOfMaxE] = max(power_rE);
        [maxValueI, indexOfMaxI] = max(power_rI);
        
        peakFreqE(k, indx) = f(indexOfMaxE);
        peakFreqI(k, indx) = f(indexOfMaxI);
        
        % Calculate the integral around the peak frequency for E population
        f_min_E = peakFreqE(indx) - bandwidth/2;
        f_max_E = peakFreqE(indx) + bandwidth/2;
        [~, f_min_idx_E] = min(abs(f - f_min_E));
        [~, f_max_idx_E] = min(abs(f - f_max_E));
        integral_peak_E(k, indx) = trapz(f(f_min_idx_E:f_max_idx_E), power_rE(f_min_idx_E:f_max_idx_E));
    
        % Calculate the integral around the peak frequency for I population
        f_min_I = peakFreqI(indx) - bandwidth/2;
        f_max_I = peakFreqI(indx) + bandwidth/2;
        [~, f_min_idx_I] = min(abs(f - f_min_I));
        [~, f_max_idx_I] = min(abs(f - f_max_I));
        integral_peak_I(k, indx) = trapz(f(f_min_idx_I:f_max_idx_I), power_rI(f_min_idx_I:f_max_idx_I));

        indx = indx + 1;
    
    end
end

% Calculate the mean and standard deviation across trials for each sample
peak_meanAcrossTrialsE = mean(peakFreqE, 1);
peak_stdAcrossTrialsE = std(peakFreqE, 0, 1);

peak_meanAcrossTrialsI = mean(peakFreqI, 1);
peak_stdAcrossTrialsI = std(peakFreqI, 0, 1);

integral_meanAcrossTrialsE = mean(integral_peak_E, 1); 
integral_stdAcrossTrialsE = std(integral_peak_E, 0, 1);  

integral_meanAcrossTrialsI = mean(integral_peak_I, 1);
integral_stdAcrossTrialsI = std(integral_peak_I, 0, 1);

figure(1)
errorbar(In_ext_I_rng, peak_meanAcrossTrialsE, peak_stdAcrossTrialsE, 'o');
hold on
errorbar(In_ext_I_rng, peak_meanAcrossTrialsI, peak_stdAcrossTrialsI, 'o');

plot(In_ext_I_rng, peak_meanAcrossTrialsE, 'b-');
plot(In_ext_I_rng, peak_meanAcrossTrialsI, 'r-');

title('Mean and Standard Deviation Peak Frequency Across Trials');
xlabel('Ext current to I')
ylabel('Peak Frequency [Hz]')
legend('E population', 'I population')
hold off

% Plot the mean value for each sample
figure(2)
errorbar(In_ext_I_rng, integral_meanAcrossTrialsE, integral_stdAcrossTrialsE, 'o');
hold on
errorbar(In_ext_I_rng, integral_meanAcrossTrialsI, integral_stdAcrossTrialsI, 'o');

plot(In_ext_I_rng, integral_meanAcrossTrialsE, 'b-');
plot(In_ext_I_rng, integral_meanAcrossTrialsI, 'r-');

hold off
title('Mean and Standard Deviation Power Integral Across Trials');
xlabel('Ext current to I');
ylabel('Power Integral');
legend('E population', 'I population')

toc;

%% functions

function [r] = simulate_WC(W, T, dt, I_ext_E, I_ext_I, NE, NI)

% Excitatory parameters
tau_E = 20;     %Timescale of the E population [ms]
a_E = 1;        %Gain of the E population
theta_E = 5;    %Threshold of the E population

% Inhibitory parameters
tau_I = 10;     %Timescale of the I population [ms]
a_I = 1;        %Gain of the I population
theta_I = 20;   %Threshold of the I population

numNeurons = length(W); 

range_t = 0:dt:T;       %Initializing time axis
N = length(range_t);    %Number of samples

r = zeros(numNeurons, N);   %Firing rate array

%Adding some noise to the input currents
Inp_ext_I = I_ext_I * (1+0.5*(rand(NE, N)-.5));
Inp_ext_E = I_ext_E * (1+0.5*(rand(NI, N)-.5));

%Solving the Wilson-Cowan equations
for i=1:N-1 

    dr_E = dt/tau_E*(-r(1:NE, i) + F(W(1:NE, :)*r(:, i) + Inp_ext_E(:,i), a_E, theta_E));
    dr_I = dt/tau_I*(-r(1+NE:end, i) + F(W(1+NE:end,:)*r(:, i) + Inp_ext_I(:,i), a_I, theta_I));

    r(1:NE,i+1) = r(1:NE,i) + dr_E;
    r(1+NE:end,i+1) = r(1+NE:end,i) + dr_I;
end

end

function f = F(x, a, theta)
    f = (1 + exp(-a * (x - theta))).^-1 - (1 + exp(a * theta)).^-1;
end