% This is an example script to instantiate an Experiment object and analyze the data.
addpath(genpath('../')) % Adding all the code
 
% First, add the path containing the data, needs to have both neural data (from Goard lab pipeline) and stimulus data, see xxxx
addpath('./data');

neural_data = importdata('neural_data.mat');
stimulus_data = importdata('stimulus_data.mat');

% Instantiate the proper Experiment object:
experiment = RSCSomaDualExperiment(neural_data, stimulus_data);

% All the necessary analyses are contained as methods within this object... 
% Feel free to use tab-completion in order to search the available methods

% For example:

% determining whether or not a cell is heading responsive
ishd = experiment.calculateHeadDirection(50); % only 50 shuffle iterations for speed

% Getting preferred directions
pd = experiment.calculatePreferredDirection('fit'); % use the 'fit' method

% Getting tuning curves
tc = experiment.calculateTuningCurves();

% From here... you're free to do anything you dream