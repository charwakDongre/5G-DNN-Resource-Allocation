function predicted_bw = xgboost_integration_function(input_data)
% XGBOOST_INTEGRATION_FUNCTION Takes simulated data, converts it to the 
% 14-feature format required by the XGBoost model, runs the prediction 
% using the Python engine, and returns the allocated bandwidth.
%
% Input: input_data (1x5 vector)
% [Signal_Strength_dBm, Latency_ms, Required_Bandwidth_Mbps, Resource_Allocation_Pct, App_Type_Index]
%
% Output: predicted_bw (scalar float)

    % Get global handles (set in res_allocation_xgboost.m)
    global XGB_MODEL; 
    global FEATURE_NAMES;
    global NP_MODULE;

    % --- 1. Map Input Data ---
    signal_strength_dBm = input_data(1);
    latency_ms = input_data(2);
    required_bw_mbps = input_data(3);
    resource_allocation_pct = input_data(4);
    app_type_index = input_data(5); % Index 1 to 10

    % --- 2. Define Application Type Mapping (MUST match simulation params) ---
    app_types = {
        'Emergency_Service', 'File_Download', 'IoT_Temperature', 'Online_Gaming', 'Streaming',
        'Video_Call', 'Video_Streaming', 'VoIP_Call', 'Voice_Call', 'Web_Browsing'
    };
    app_name = app_types{app_type_index};
    
    % --- 3. Build 1x14 Feature Vector (One-Hot Encoding) ---
    num_features = length(FEATURE_NAMES);
    input_vector = zeros(1, num_features);

    % Map continuous features to the first 4 slots
    input_vector(1) = signal_strength_dBm;
    input_vector(2) = latency_ms;
    input_vector(3) = required_bw_mbps;
    input_vector(4) = resource_allocation_pct;
    
    % Handle One-Hot Encoding for Application Type
    % The feature name starts with 'App_' and is followed by the type (e.g., 'App_Streaming')
    app_col_name = ['App_', app_name];
    
    % Find the index of the specific App_ column in the globally stored feature list
    idx = find(strcmp(FEATURE_NAMES, app_col_name), 1);
    
    if ~isempty(idx) && idx > 4 % Check if the feature name exists and is not one of the first four (continuous)
        input_vector(idx) = 1; % Set the correct application column to 1
    end

    % --- 4. Prediction via Python ---
    
    % Convert the 1x14 MATLAB vector to a NumPy array (required by XGBoost)
    arr = NP_MODULE.array(input_vector, 'float64').reshape(int32(1), int32(num_features));
    
    % Call the predict method on the global XGBoost model
    py_pred = XGB_MODEL.predict(arr);
    
    % Convert Python result (NumPy array with 1 element) to MATLAB double
    predicted_bw = double(py_pred{1}); 

end