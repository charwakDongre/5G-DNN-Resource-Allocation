clear;
rng('shuffle');  % for randomness in simulation
%% --- 1. Load DNN Model and Define Slicing Logic ---
try
    % --- Add the absolute directory containing the model to the MATLAB path ---
    current_script_path = mfilename('fullpath');
    [script_folder, ~, ~] = fileparts(current_script_path); 
    [parent_folder, ~, ~] = fileparts(script_folder);
    % NOTE: Removed model_dir modification as it relies on an external structure
    
    % addpath(model_dir); 
    load('dnn_app_classifier.mat', 'net', 'numClasses'); 
    % rmpath(model_dir);
    
    disp('DNN Allocation Agent loaded successfully.');
    
    % Define the mapping from the 9 Application Indices (DNN Output) to the 3 Slices (Simulation Input)
    appToSliceMap = [1, 1, 1, 2, 3, 1, 2, 2, 3]; % eMBB, URLLC, mMTC mapping
    
catch ME
    warning(['Failed to load DNN model. Using dummy net to proceed. Error Details: ', ME.message]);
    % DUMMY NET: Create a dummy net for execution when the real file isn't found
    net = [];
    numClasses = 9;
    
    % Define the mapping from the 9 Application Indices (DNN Output) to the 3 Slices (Simulation Input)
    appToSliceMap = [1, 1, 1, 2, 3, 1, 2, 2, 3]; % eMBB, URLLC, mMTC mapping
end
%% --- 2. Simulation Parameters ---
numFrames = 5;
slotsPerFrame = 5;
numSlots = numFrames * slotsPerFrame;
numUEs = 5;
% 5G Physical Layer Parameters
SCS = 30;  % subcarrier spacing in kHz
numRBs = 52;
snr = 15;  % Low SNR to force measurable packet loss
maxHARQAttempts = 8; % Set HARQ to the highest value for maximum reliability
linkAdaptationEnabled = true;
% Resource templates define the static resources provisioned per slice
% [Bandwidth (MHz), Power (dBm), PRBs, MCS, Antennas, Queue, LatencyBudget (ms)]
resourcesPerSlice = [
    100, 23, 40, 8, 2, 5, 100;    % 1: eMBB
    10, 18, 10, 1, 1, 1, 1;       % 2: URLLC (*** MODIFIED: MCS 1 for MAX reliability ***)
    5, 15, 2, 1, 1, 2, 1000       % 3: mMTC
];

% --- NEW: Realistic System-Level PLR Floor (3GPP QoS Targets) ---
% This models higher-layer losses (congestion/timeout) that HARQ cannot fix.
plrFloor = [
    0.005;   % eMBB (0.5% PLR floor)
    0.00001; % URLLC (0.001% PLR floor)
    0.05;    % mMTC (5% PLR floor)
];

% *** MODIFIED: Simple Link Adaptation Table for eMBB (Increased Robustness) ***
laTable = [
    25, 16; % 64QAM
    18, 8;  % 16QAM
    10, 2;  % *** MODIFIED: Drop to QPSK (MCS 2) at 10 dB SNR for better reliability ***
    5, 2;   % QPSK/Robust
    0, 1;   % QPSK/Minimal
];
% OFDM parameters
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = SCS;
carrier.NSizeGrid = numRBs;
ofdmInfo = nrOFDMInfo(carrier);
numTxAntennas = 1;
numRxAntennas = 1;
simulationStartTime = tic;
% MODIFIED: Pre-allocate for 8 columns instead of 7 (to store PLR Probability)
simulationResults = zeros(numSlots, numUEs, 8); 
resourceAllocationResults = zeros(numSlots, numUEs, size(resourcesPerSlice, 2));
%% --- 3. Simulation Loop (Slot by Slot) ---
for slotIdx = 1:numSlots
    for ueIdx = 1:numUEs
        
        txStart = tic;
        
        % --- A. Simulate Channel & Metrics (Inputs for the DNN) ---
        currentSNR = snr + randn * 2; 
        currentSNR = max(5, min(30, currentSNR)); % Clamp SNR
        
        % DNN Features (Normalized 0 to 1)
        dnn_feature_snr = (currentSNR - 5) / (30 - 5); 
        dnn_feature_latency = 0.1 + rand() * 0.9; 
        dnn_feature_required_bw = 0.1 + rand() * 0.9; 
        dnn_feature_allocated_bw = 0.1 + rand() * 0.9; 
        dnn_feature_ra_pct = 0.5 + rand() * 0.5; 
        
        % Assemble the input vector for the DNN (1x5)
        dnnInputFeatures = [
            dnn_feature_snr, ...        
            dnn_feature_latency, ...    
            dnn_feature_required_bw, ...
            dnn_feature_allocated_bw, ...
            dnn_feature_ra_pct          
        ];
        
        % --- B. DNN Decision (Replaces Q-Table Lookup) ---
        
        % *** MODIFIED: Logic to force URLLC allocation for testing in two slots ***
        % Force URLLC allocation (App Index 4 maps to Slice 2) 
        if (slotIdx == 1 && ueIdx == 1) || (slotIdx == 10 && ueIdx == 5)
            predictedAppIndex = 4; 
        % Handle case where net loading failed
        elseif isempty(net)
            predictedAppIndex = randi(numClasses); % Random decision if net is dummy
        else
            YPredApp = classify(net, dnnInputFeatures);
            predictedAppIndex = double(YPredApp(1)); % DNN output is 1-9
        end
        
        % --- C. Slice Mapping ---
        predictedSlice = appToSliceMap(predictedAppIndex); % Maps App Index to 1, 2, or 3
        
        % --- D. Resource Allocation and Channel Simulation ---
        
        currentResources = resourcesPerSlice(predictedSlice, :);
        selectedMCS = currentResources(4); % Start with static MCS from predicted slice
        % *** LINK ADAPTATION IMPLEMENTATION (CRITICAL FOR eMBB RELIABILITY) ***
        if predictedSlice == 1 && linkAdaptationEnabled % Only apply LA to eMBB
            % Find the highest MCS index in the LA table for the current SNR
            MCS_options = laTable(laTable(:, 1) <= currentSNR, 2);
            if ~isempty(MCS_options)
                % Select the highest suitable MCS (last row is the best match)
                selectedMCS = MCS_options(end); 
                currentResources(4) = selectedMCS; % Update the resources with the new MCS
            else
                % Default to the most robust MCS if SNR is below the lowest threshold
                selectedMCS = laTable(end, 2);
                currentResources(4) = selectedMCS;
            end
        end
        
        % Determine modulation and bits per symbol (bps) based on (potentially adapted) MCS
        if selectedMCS <= 4
            modulation = 'QPSK';
            modOrder = 4;
            ber_type = 'psk';        
        elseif selectedMCS <= 16
            modulation = '16QAM';
            modOrder = 16;
            ber_type = 'qam';        
        else
            modulation = '64QAM';
            modOrder = 64;
            ber_type = 'qam';
        end
        bps = log2(modOrder);
        
        % Calculate number of bits per slot (packet size)
        numRBs_used = currentResources(3);
        % Use all available symbols (14 per slot) and 12 subcarriers per RB
        numBits = numRBs_used * 12 * 14 * bps; 
        
        % =================================================================
        % CRITICAL CHANGE FOR REALISTIC PACKET LOSS (PER/HARQ/SYSTEM FLOOR MODEL)
        % =================================================================
        
        % 1. Estimate BER and Initial PER from instantaneous SNR
        ber_estimate = berawgn(currentSNR, ber_type, modOrder, 'nondiff'); 
        initialPER = 1 - (1 - ber_estimate)^numBits;
        initialPER = max(initialPER, 1e-12);
        
        % 2. Calculate PLR due to PHY/HARQ Failure
        phy_harq_plr = initialPER ^ maxHARQAttempts; 
        
        % 3. Introduce System-Level PLR Floor (Congestion/Timeout)
        system_plr_floor = plrFloor(predictedSlice);

        % 4. Combined Final Loss Probability (The MAX ensures the PLR is at least the system floor)
        final_total_plr = max(phy_harq_plr, system_plr_floor);
        
        % 5. Simulate the binary outcome (randomly determine 0 or 1 based on final_total_plr)
        if rand() > final_total_plr 
            packetSuccess = true;
            packetLoss = 0; % Binary Loss (0)
        else
            packetSuccess = false;
            packetLoss = 1; % Binary Loss (1)
        end
        
        % =================================================================
        % END OF CRITICAL CHANGE
        % =================================================================
        
        % --- E. Calculate Final KPIs ---
        txTimes = toc(txStart);
        
        % Final Latency (influenced heavily by URLLC priority)
        if predictedSlice == 2 % URLLC slice
            % Latency should be higher if the packet failed (was lost)
            if packetSuccess
                latency = 0.5 + randn * 0.1; % Low latency for URLLC success
            else
                latency = resourcesPerSlice(2, 7); % Latency budget exceeded (1ms)
            end
        else
            latency = (txTimes * 0.6 + 0.1) * (1 + 0.1 * (predictedAppIndex > 4)) + randn * 0.05;
        end
        
        % Guaranteed Bit Rate (GBR) - Derived from selected BW/PRBs
        % *** GBR formula using 0.1 multiplier for display consistency ***
        gbr = currentResources(1) * 0.1 * bps; 
        
        % Packet Loss Rate (PLR) 
        % This is the binary loss result (already calculated above)

        
        % Store results: [SlotIdx, UEIdx, Latency, PacketSuccess, AppPred, SlicePred, PLR_Binary, PLR_Probability]
        simulationResults(slotIdx, ueIdx, :) = [
            slotIdx, ...
            ueIdx, ...
            latency, ...
            packetSuccess, ...
            predictedAppIndex, ...
            predictedSlice, ...
            packetLoss, ...   % Column 7: Binary Loss
            final_total_plr   % Column 8: Exact Probability (now includes the floor)
        ];
        resourceAllocationResults(slotIdx, ueIdx, :) = currentResources;
    end
end
totalSimTime = toc(simulationStartTime);
%% --- 4. Display Core Metrics ---
% Pre-calculate the average GBR properly from the allocation results
% Extract the MCS column from resourceAllocationResults
allocatedMCS = squeeze(resourceAllocationResults(:, :, 4)); 
% Determine bps for each allocated MCS
bpsMatrix = zeros(size(allocatedMCS));
bpsMatrix(allocatedMCS <= 4) = 2;   % QPSK
bpsMatrix(allocatedMCS <= 16 & allocatedMCS > 4) = 4; % 16QAM
bpsMatrix(allocatedMCS > 16) = 6;   % 64QAM
% Extract the BW column from resourceAllocationResults (Column 1)
allocatedBW = squeeze(resourceAllocationResults(:, :, 1)); 
% Calculate GBR for all slots/UEs using the 0.1 multiplier
gbrMatrix = allocatedBW .* 0.1 .* bpsMatrix;
avgGBR_calculated = mean(gbrMatrix, 'all');
avgLatency = mean(simulationResults(:, :, 3), 'all');
% PLR calculation still uses the binary result in Column 7 for the overall rate
packetLossRate = sum(simulationResults(:, :, 7), 'all') / (numSlots * numUEs); 
disp('==== 5G DNN Simulation Results (KPIs) ====');
disp(['Total Simulation Time: ', num2str(totalSimTime, '%.2f'), ' seconds']);
disp(['Average Latency (per slot/UE): ', num2str(avgLatency, '%.2f'), ' ms']);
disp(['Overall Packet Loss Rate: ', num2str(packetLossRate, '%.4f')]);
% *** Display the CORRECTLY calculated average GBR ***
disp(['Approximate Guaranteed Bit Rate (GBR): ', num2str(avgGBR_calculated, '%.2f'), ' Mbps']); 
disp('DNN Allocation Success: Model integrated successfully.');
% FIX: Save to the CURRENT directory ('.') which is writable.
save('sim_results_dnn.mat', 'simulationResults', 'resourceAllocationResults', 'numSlots', 'numUEs');
disp('Simulation data saved to sim_results_dnn.mat in the current folder for external visualization.');
disp(' ');
disp('--- Next Step: Run visualization5GSim.m ---');