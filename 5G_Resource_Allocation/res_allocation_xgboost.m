clear;
rng('shuffle');  % for randomness in simulation

% --- GLOBAL VARIABLES FOR PYTHON/MATLAB INTEGRATION ---
global XGB_MODEL; 
global FEATURE_NAMES;
global NP_MODULE;

% -------------------------------------------------------------------------
% 1. PYTHON SETUP (CRITICAL STEP - FINAL PYTHONHOME FIX)
% -------------------------------------------------------------------------

% --- Miniconda Root Path (Directory containing 'envs' folder) ---
% NOTE: This path points to the environment folder, NOT the python.exe itself.
PYTHON_ROOT_DIR = 'C:\\Users\\Windows\\miniconda3\\envs\\matlab_xgboost';

try
    % 1. Force Python engine to unload any old/stuck configurations
    pyenv('Version', ''); % Clears the broken '/usr/bin/python3' configuration
    
    % 2. Set the PYTHONHOME environment variable for this MATLAB session
    setenv('PYTHONHOME', PYTHON_ROOT_DIR);
    
    % 3. Try to initialize the engine again (MATLAB will now look at PYTHONHOME)
    pe = pyenv;
    
    % Verification check
    if pe.Status ~= "Loaded"
        error('MATLAB failed to link to Miniconda environment.');
    end
    
    disp('Python environment successfully loaded from PYTHONHOME.');

catch ME
    error(['FATAL PYTHON SETUP ERROR: Cannot link using PYTHONHOME.\n',...
           '1. Path Attempted: %s\n',...
           '2. Error: %s'], PYTHON_ROOT_DIR, ME.message);
end

% Load Python modules
try
    % Since the link succeeded, the modules should be found.
    xgb = py.importlib.import_module('xgboost');
    np = py.importlib.import_module('numpy');
    sio = py.importlib.import_module('scipy.io');
    
    % Set global NumPy module handle
    global NP_MODULE;
    NP_MODULE = np; 

catch ME
    error(['MODULE LOAD ERROR: Dependencies missing.\n',...
           'Ensure xgboost, pandas, and scipy are installed in the matlab_xgboost environment.\n',...
           'Error: %s'], ME.message);
end

% -------------------------------------------------------------------------
% 2. LOAD MODEL AND FEATURE NAMES
% -------------------------------------------------------------------------
try
    XGB_MODEL = xgb.XGBRegressor();
    XGB_MODEL.load_model("xgboost_model.json");
    feature_data = sio.loadmat('feature_names.mat');
    FEATURE_NAMES = cell(feature_data.feature_names); 
    disp('XGBoost model and features loaded successfully.');
catch ME
    error('MODEL FILE ERROR: Ensure xgboost_model.json and feature_names.mat are in the current MATLAB folder. Error: %s', ME.message);
end

% -------------------------------------------------------------------------
% 3. SIMULATION PARAMETERS
% -------------------------------------------------------------------------
numFrames = 10;
slotsPerFrame = 10;
numSlots = numFrames * slotsPerFrame;
numUEs = 5;

% Define all possible application types (MUST match one-hot encoding!)
app_types = {
    'Emergency_Service', 'File_Download', 'IoT_Temperature', 'Online_Gaming', 'Streaming',
    'Video_Call', 'Video_Streaming', 'VoIP_Call', 'Voice_Call', 'Web_Browsing'
};
numAppTypes = length(app_types);

% Results storage: [SlotIdx, UEIdx, Latency, PacketSuccess, AppTypeIndex, Predicted_BW, PacketLoss]
simulationResults = zeros(numSlots, numUEs, 7); 

% Pre-assign consistent application types for the 5 UEs
ueAppTypeIndex = randi([1, numAppTypes], 1, numUEs);

simulationStartTime = tic;

% -------------------------------------------------------------------------
% 4. SIMULATION LOOP
% -------------------------------------------------------------------------

for slotIdx = 1:numSlots
    for ueIdx = 1:numUEs

        % --- A. SIMULATE CHANNEL AND USER STATE (Features) ---
        
        currentAppIndex = ueAppTypeIndex(ueIdx);

        % Simulate Signal/Channel Quality (Signal_Strength_dBm)
        snr = 15 + randn * 3; % Base SNR 15 dB
        signal_strength_dBm = -80 + snr * 2; % Lower dBm is worse
        
        % Simulate Network Load/Congestion (Resource_Allocation_Pct)
        resource_alloc_pct = 0.5 + 0.4 * rand();
        
        % Simulate Latency (Latency_ms)
        latency_ms = 10 + randn * 5; 
        
        % Simulate Required Bandwidth (Based loosely on App Type)
        required_bw_mbps = (currentAppIndex/2) + 1 + randn * 0.5;
        required_bw_mbps = max(0.1, required_bw_mbps); 

        % Simulate Packet Loss (For visualization metrics)
        packetLoss = 0.05 + 0.03 * randn(); 
        packetLoss = min(max(packetLoss, 0.01), 0.1);

        % --- B. PREDICT ALLOCATION USING XGBOOST ---
        
        % XGBoost Input Vector:
        % [Signal_Strength_dBm, Latency_ms, Required_Bandwidth_Mbps, Resource_Allocation_Pct, App_Type_Index]
        input_data = [
            signal_strength_dBm, 
            latency_ms, 
            required_bw_mbps, 
            resource_alloc_pct, 
            currentAppIndex
        ];
        
        % Call the prediction function defined in xgboost_integration_function.m
        predicted_bw = xgboost_integration_function(input_data);
        
        % --- C. STORE RESULTS ---
        
        % simulationResults: [SlotIdx, UEIdx, Latency, PacketSuccess, AppTypeIndex, Predicted_BW, PacketLoss]
        simulationResults(slotIdx, ueIdx, :) = [
            slotIdx, 
            ueIdx, 
            latency_ms, 
            1, % Assume success if BW is allocated
            currentAppIndex, 
            predicted_bw, 
            packetLoss
        ];
    end
end
totalSimTime = toc(simulationStartTime);

% Save results for visualization
save('sim_results_xgboost.mat', 'simulationResults', 'numSlots', 'numUEs', 'app_types', 'totalSimTime');

% -------------------------------------------------------------------------
% 5. DISPLAY SUMMARY METRICS
% -------------------------------------------------------------------------
disp('==== XGBoost Allocation Simulation Results ====');
disp(['Total Simulation Time: ', num2str(totalSimTime, '%.2f'), ' seconds']);

% Metrics calculated over the entire simulation
avgLatency = mean(simulationResults(:, :, 3), 'all');
avgAllocatedBW = mean(simulationResults(:, :, 6), 'all');
avgPacketLoss = mean(simulationResults(:, :, 7), 'all');

disp(['Average Latency (Overall): ', num2str(avgLatency, '%.2f'), ' ms']);
disp(['Average Allocated Bandwidth: ', num2str(avgAllocatedBW, '%.2f'), ' Mbps']);
disp(['Average Packet Loss Rate: ', num2str(avgPacketLoss, '%.4f')]);

disp('\nRun visualization5GSim_xgboost.m to generate plots.');