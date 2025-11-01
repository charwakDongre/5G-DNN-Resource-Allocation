% predict_in_matlab.m

% -------------------------------------------------------------------------
% 1. PYTHON SETUP (CRITICAL STEP - FINAL STABLE FIX)
% -------------------------------------------------------------------------

% We rely on MATLAB's default environment detection (which falls back to the system Python).
try
    % Attempt to set the version to empty, which triggers MATLAB's default search.
    pyenv('Version', ''); 
catch ME
    error(['PYTHON ENVIRONMENT ERROR: MATLAB cannot initialize Python.\n',...
           'Check system MATLAB configuration and installation. Error: %s'], ME.message);
end

% Check which Python executable MATLAB settled on
current_pe = pyenv; 

% Load necessary Python modules into the MATLAB workspace
try
    % ------------------------------------------------------------------
    % ** MODULE LOCATION FIX (The Final Solution for Mac)**
    % The user's pip install put packages in a user-specific folder.
    % We must explicitly add this path to Python's search path (sys.path).
    
    % Get the MATLAB user home directory 
    user_home = char(py.os.path.expanduser('~'));
    
    % --- Dynamic User Site-Package Path Check ---
    % Since the system Python (e.g., 3.10) can't see modules installed via 
    % user pip (which often targets Python 3.9 or 3.10), we append known locations.
    
    % List of common user site-package paths on macOS (using Python 3.9/3.10 standards)
    potential_site_packages = {
        fullfile(user_home, 'Library', 'Python', '3.9', 'lib', 'python', 'site-packages'),
        fullfile(user_home, 'Library', 'Python', '3.10', 'lib', 'python', 'site-packages'),
        fullfile(user_home, 'Library', 'Python', '3.11', 'lib', 'python', 'site-packages') % Added 3.11 just in case
    };
    
    % Append all potential paths to Python's system search path
    for i = 1:length(potential_site_packages)
        current_site_path = potential_site_packages{i};
        if exist(current_site_path, 'dir') % Only append if the folder actually exists
            py.sys.path.append(current_site_path);
        end
    end
    
    % ------------------------------------------------------------------
    
    % Now attempt to import modules (Python should look in appended paths)
    xgb = py.importlib.import_module('xgboost');
    np = py.importlib.import_module('numpy');
    sio = py.importlib.import_module('scipy.io');
catch ME
    % The modules should now be installed in the environment that MATLAB settled on.
    error(['FINAL PYTHON MODULE ERROR: The Python environment is missing XGBoost/NumPy/SciPy.\n',...
           '1. MATLAB successfully loaded executable: %s\n',...
           '2. The script failed trying to load modules. Please ensure your Python installation is complete and accessible, and restart MATLAB.'], char(current_pe.Executable));
end

% -------------------------------------------------------------------------
% 2. LOAD MODEL AND FEATURE NAMES
% -------------------------------------------------------------------------
try
    model = xgb.XGBRegressor();
    model.load_model("xgboost_model.json"); % Load the saved XGBoost model
    
    feature_data = sio.loadmat('feature_names.mat');
    % Convert the Python-compatible NumPy array of names into a MATLAB cell array
    feature_names = cell(feature_data.feature_names); 
    num_features = length(feature_names);
catch ME
    error('MODEL FILE ERROR: Ensure xgboost_model.json and feature_names.mat are in the current MATLAB folder. Error: %s', ME.message);
end

% -------------------------------------------------------------------------
% 3. DEFINE A NEW USER SCENARIO (TEST CASE)
% -------------------------------------------------------------------------
% Features: 1. Signal_Strength_dBm, 2. Latency_ms, 3. Required_Bandwidth_Mbps, 
%           4. Resource_Allocation_Pct, 5. Application_Type
new_user_inputs = { -75, 30, 10, 0.7, 'Video_Call' }; 
input_feature_names = {'Signal_Strength_dBm', 'Latency_ms', 'Required_Bandwidth_Mbps', 'Resource_Allocation_Pct', 'Application_Type'};

% -------------------------------------------------------------------------
% 4. FEATURE ALIGNMENT AND ONE-HOT ENCODING
% -------------------------------------------------------------------------
input_vector = zeros(1, num_features);

for i = 1:length(input_feature_names)
    feature_name = input_feature_names{i};
    input_value = new_user_inputs{i};
    
    if strcmp(feature_name, 'Application_Type')
        % Handle One-Hot Encoding: Find the corresponding 'App_' column
        app_col_name = ['App_', input_value];
        idx = find(strcmp(feature_names, app_col_name), 1);
        if ~isempty(idx)
            input_vector(idx) = 1; % Set the correct application column to 1
        end
    else
        % Handle Numerical Features: Find the numerical column
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
    disp('DECISION: Excellent quality of service and low risk of degradation.');
else
    disp('INTERPRETATION: Allocation meets required QoS precisely.');
    disp('DECISION: Service quality is maintained as demanded.');
end

fprintf('------------------------------------------------\n');
