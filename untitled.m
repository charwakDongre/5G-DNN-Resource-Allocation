% predict_in_matlab.m

% -------------------------------------------------------------------------
% 1. PYTHON SETUP (CRITICAL STEP)
%    Replace the path below with the EXACT path to your Python executable
%    (e.g., /usr/local/bin/python3 or /opt/homebrew/bin/python3)
% -------------------------------------------------------------------------
pyenv('Version','/Library/Frameworks/Python.framework/Versions/3.12/bin/python3'); 

% Load necessary Python modules into the MATLAB workspace
xgb = py.importlib.import_module('xgboost');
np = py.importlib.import_module('numpy');
sio = py.importlib.import_module('scipy.io');

% -------------------------------------------------------------------------
% 2. LOAD MODEL AND FEATURE NAMES
%    These files must be in the same folder as this script.
% -------------------------------------------------------------------------
try
    model = xgb.XGBRegressor();
    model.load_model("xgboost_model.json");
    
    feature_data = sio.loadmat('feature_names.mat');
    % Convert the Python-compatible NumPy array of names into a MATLAB cell array
    feature_names = cell(feature_data.feature_names); 
    num_features = length(feature_names);
catch ME
    error('Model/File Loading Error: Ensure xgboost_model.json and feature_names.mat are in the current MATLAB folder, and pyenv is set correctly. Error: %s', ME.message);
end

% -------------------------------------------------------------------------
% 3. DEFINE A NEW USER SCENARIO (TEST CASE)
%    Change these values to test different scenarios.
% -------------------------------------------------------------------------
% Features: 1. Signal_Strength_dBm, 2. Latency_ms, 3. Required_Bandwidth_Mbps, 
%           4. Resource_Allocation_Pct, 5. Application_Type
new_user_inputs = { -75, 30, 10, 0.7, 'Video_Call' }; 
input_feature_names = {'Signal_Strength_dBm', 'Latency_ms', 'Required_Bandwidth_Mbps', 'Resource_Allocation_Pct', 'Application_Type'};

% -------------------------------------------------------------------------
% 4. FEATURE ALIGNMENT AND ONE-HOT ENCODING
%    This block maps the 5 user inputs into the 14-column vector expected 
%    by the trained XGBoost model.
% -------------------------------------------------------------------------
input_vector = zeros(1, num_features);

for i = 1:length(input_feature_names)
    feature_name = input_feature_names{i};
    input_value = new_user_inputs{i};
    
    if strcmp(feature_name, 'Application_Type')
        % Handle One-Hot Encoding (converts 'Video_Call' to 'App_Video_Call')
        app_col_name = ['App_', input_value];
        idx = find(strcmp(feature_names, app_col_name), 1);
        if ~isempty(idx)
            input_vector(idx) = 1; % Set the correct application column to 1
        end
    else
        % Handle Numerical Features (maps numeric value to correct index)
        idx = find(strcmp(feature_names, feature_name), 1);
        if ~isempty(idx)
            input_vector(idx) = input_value;
        end
    end
end

% -------------------------------------------------------------------------
% 5. PREDICT RESOURCE ALLOCATION
% -------------------------------------------------------------------------

% Convert the MATLAB input vector to a Python NumPy array
arr = np.array(input_vector, 'float64').reshape(int32(1), int32(num_features));

% Call the Python model's predict method
py_pred = model.predict(arr);

% Convert the Python result back to a standard MATLAB double
allocated_bw = double(py_pred{1});

% -------------------------------------------------------------------------
% 6. DISPLAY RESULTS AND INTERPRETATION
% -------------------------------------------------------------------------
fprintf('\n------------------------------------------------\n');
fprintf('  5G RESOURCE ALLOCATION PREDICTION\n');
fprintf('------------------------------------------------\n');
fprintf('Tested Application: %s\n', new_user_inputs{5});
fprintf('Required Bandwidth: %.2f Mbps\n', new_user_inputs{3});
fprintf('Predicted Allocation: %.2f Mbps\n', allocated_bw);
fprintf('------------------------------------------------\n');

% Interpretation
required_bw = new_user_inputs{3};
if allocated_bw < required_bw
    disp('INTERPRETATION: Allocation is insufficient. Network resources are likely constrained, or channel quality is poor relative to demand.');
    disp('DECISION: User may experience degradation in service (e.g., poor video quality).');
elseif allocated_bw > required_bw * 1.1 % Simple buffer check
    disp('INTERPRETATION: Allocation exceeds requirement. This ensures high QoS and accounts for potential fluctuations.');
    disp('DECISION: Excellent service quality and low risk of degradation.');
else
    disp('INTERPRETATION: Allocation meets required QoS precisely.');
    disp('DECISION: Service quality is maintained as demanded.');
end

fprintf('------------------------------------------------\n');
