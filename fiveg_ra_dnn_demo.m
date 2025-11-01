%% 5G Application Type Classification Demo (Using Real QoS Data)
clc; clear; close all;


outDir = 'outputs';
[success, message, messageId] = mkdir(outDir);
if ~exist(outDir, 'dir')
    warning('Error creating output directory: %s. Continuing without saving plots/model.', message);
    outDir = ''; % Disable saving if folder creation failed
end


numFeatures = 5;    % Fixed input features: Signal, Latency, Required/Allocated Bandwidth, Pct
numClasses  = 9;    % The number of Application Types (Output classes)

% Define the names corresponding to the 1-based indices for final output
APP_NAMES = {'Video_Call', 'Web_Browsing', 'Streaming', ...
             'Emergency_Service', 'Background_Download', 'Video_Streaming', ...
             'VoIP_Call', 'Online_Gaming', 'IoT_Temperature'};

%% Load Pre-Processed and Scaled Data
% Data is loaded from the CSV files created in the previous step.
try
    XTrain_raw = readmatrix('XTrain_scaled.csv');
    YTrain_raw = readmatrix('YTrain_codes.csv');
    XVal_raw   = readmatrix('XVal_scaled.csv');
    YVal_raw   = readmatrix('YVal_codes.csv');
    XTest_raw  = readmatrix('XTest_scaled.csv');
    YTest_raw  = readmatrix('YTest_codes.csv');
catch ME
    error('File Loading Error: Could not find or read one of the processed CSV files. Please ensure all 6 files (X/Y Train/Val/Test) are in the current folder. MATLAB Error: %s', ME.message);
end

% Convert labels to categorical (Y_raw + 1 converts 0-indexed codes to 1-based indices)
YTrain = categorical(YTrain_raw + 1);
YVal   = categorical(YVal_raw + 1);
YTest  = categorical(YTest_raw + 1);

% Ensure X data is double type as expected by DNN
XTrain = double(XTrain_raw);
XVal   = double(XVal_raw);
XTest  = double(XTest_raw);

fprintf('Dataset Loaded: Train=%d, Validation=%d, Test=%d\n', length(YTrain), length(YVal), length(YTest));
fprintf('Classification Target: %d Application Types\n', numClasses);

%% Define Custom Class Weights 

classWeights = [4 6 8 7 5 4 8 5 7]; 

%% Define DNN Model (Architecture updated for 9 classes and capacity)
layers = [
    featureInputLayer(numFeatures)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(256) 
    reluLayer
    fullyConnectedLayer(numClasses) 
    softmaxLayer
    % PASS WEIGHTS to the classification layer
    classificationLayer('Classes', unique(YTrain), 'ClassWeights', classWeights)];

%% Training Options (Optimized for Stability and Convergence)
numEpochs = 100; 
options = trainingOptions('adam', ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.005, ...         
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.5, ...        
    'LearnRateDropPeriod', 25, ...         
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

%% Train Network Manually
fprintf('\n--- Starting DNN Training on 5G QoS Data ---\n');
trainingAccuracy   = zeros(numEpochs,1);
validationAccuracy = zeros(numEpochs,1);
net = [];

for epoch = 1:numEpochs
    net = trainNetwork(XTrain, YTrain, layers, options);
    
    YPredTrain = classify(net, XTrain);
    trainingAccuracy(epoch) = mean(YPredTrain == YTrain);
    
    YPredVal = classify(net, XVal);
    validationAccuracy(epoch) = mean(YPredVal == YVal);
    
    if mod(epoch, 10) == 0 || epoch == 1
        fprintf("Epoch %d/%d: Train Acc = %.2f%%, Val Acc = %.2f%%\n", ...
            epoch, numEpochs, trainingAccuracy(epoch)*100, validationAccuracy(epoch)*100);
    end
end

%% Save Model 
if ~isempty(outDir)
    try
        save(fullfile(outDir, 'dnn_app_classifier.mat'), 'net', 'numClasses');
        fprintf('\nModel saved to %s/dnn_app_classifier.mat\n', outDir);
    catch ME
        warning('Failed to save model: %s', ME.message);
    end
end

%% Evaluate on Test Set (Calculation and Print prioritized)
YPredTest = classify(net, XTest);
testAcc = mean(YPredTest == YTest);

fprintf('\n--- Evaluation on Test Set ---\n');
fprintf('Final Test Accuracy (Application Classification): %.2f%%\n', testAcc*100);

%% Confusion Matrix (Calculation kept, plotting removed)
cm = confusionchart(YTest, YPredTest);
fprintf('Confusion Matrix Generated. Check model performance data.\n');


%% Fairness Analysis (Allocation Counts)
allocCounts = countcats(YPredTest);
fprintf('\n--- Fairness Analysis (Predicted Allocation Counts) ---\n');
% Print the Application Name instead of the index number
for i = 1:numClasses
    % Use the defined APP_NAMES array for clarity
    fprintf('%-20s: %d allocations\n', APP_NAMES{i}, allocCounts(i));
end
