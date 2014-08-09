%% GP_F1_head2head.m
%% 7/14/14
%  Trains GPs for each of several class labels and outputs results of
%  a head-to-head comparison. The first classifier picks which points it
%  filters the test set down to, and these are given to the other
%  classifiers for prediction.
%  This version uses the F1 score (precision*recall)/(precision+recall) as
%  a confidence metric.
%%
function GP_F1_head2head_isokd(crowdsourceCutoff,energycutoff,binThreshold)

%me = mfilename;
me = sprintf('GP_acc_phy_%d_bin_%d',crowdsourceCutoff*100,binThreshold);
use_timestamp = true;
thesisStartup;

% Hard-coded for printing convenience
metricNames = 'PointArng TriSize Extension Spread Limit PerpSym ParallelSym OrthoNorm Volume GraspIt_Volume GraspIt_Epsilon';

% Suppress pesky LR warnings
orig_warn_state = warning;
warning('off','all');

% Number of features to read
dimensionCount = 11;

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter
results_voting = [];
groundtruth = [];

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
results1str = {};
results2str = {};
res2index = [];


%Shake data collected Jan13
dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'zhifei_human_data_post-correction.csv';
end

dzzf_autograsp = 0;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_post-correction.csv';
end
   
%------------End Defind Parameters-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Seed Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////


for i = 1:length(datastr)
      datatemp = importdata(datastr{i});
      
      % For these datasets we only want the first 11 metrics
      data = [data;datatemp.data(:,1:dimensionCount)];
      % GraspIt estimate
      results1 = [results1;datatemp.data(:,dimensionCount+1)];
      % No crowdsourced data in the sets
      results_voting = [results_voting;datatemp.data(:,dimensionCount+2)];
      %results2 = ...
      % Physical testing
      groundtruth = [groundtruth;datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
%      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
%         error('headers do not match, check data files')   
      end   
end

%------------End Seed Data-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Condition Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////

%Remove NAN columns
datatemp = isnan(mean(data));
data(:,datatemp) = [];
disp('!!!!below quality measures have been removed due to insufficient data!!!!!!!!!!!!')
disp( header(datatemp))
header(datatemp) = [];
%% We want to keep all dimensions so don't delete any
%Figure out which QM's are significant with T-tests
% [p,keep,delete] = ttester(data,groundtruth,qmset,cutoff);
% disp('Keeping QMs:')
% disp('name        p values')
% for i = 1:length(keep)
%     disp([header(keep(i)) num2str(p(keep(i)),3)] )
% end
% 
% 
% %Delete non-important QM's
% disp('Deleting QMs:')
% disp('name        p values')
% for i = length(delete):-1:1
%     disp([header(delete(i)) num2str(p(delete(i)),3)] )
% end
% header(delete) = [];
% data(:,delete) = [];

data = sphereize(data);

% Run PCA on a copy for plotting the contour
finalDimensionNum = 2;
[~,data_pca,~] = pca(data);

%Reduce components based on earlier specifications
data_pca = data_pca(:,logical([ones(1,finalDimensionNum) zeros(1,size(data,2)-finalDimensionNum)]));

% Testing only on physical testing successes
physicalCutoff = .8;

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 30;

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .33;
validationProportion = .33;
%train2Proportion = .25;

filter_with_confidence = 1;

% Number of nodes to record information for on each iteration
nodes_to_keep = 3;

% A node must meet this precision value to be colored in the contour map
contourPrecisionCutoff = .5;

% A bin needs at least this many points to be considered confident
support_threshold = 3;
disp(strcat('binThreshold:',num2str(binThreshold)));
disp(strcat('crowdsource cutoff:',num2str(crowdsourceCutoff)));
disp(strcat('using confidence:',num2str(filter_with_confidence)));

%% Build global kdtree
distances = L2_distance(data',data',1);
opts = [];
opts.verbose = false;
opts.dims = 2;
opts.display = false;
if ~exist('k','var')
    % found 6 to be the best k value for the 522 datapoints
    k = 6;
end
[Y, ~, ~] = Isomap(distances,'k',k,opts);
plotted_indices = (Y.index)';
data_iso = Y.coords{1}';
data = data(plotted_indices,:);
data_pca = data_pca(plotted_indices,:);
groundtruth = groundtruth(plotted_indices);
    
tree = kdMedCreateTree(data_iso,binThreshold);
% Don't split on results, but include them for easy bin confidence
% computing
bins = kdGetAllBins(tree);
fprintf('Number of bins: %d\n',size(bins,1));
bin_size_sum = 0;
for i = 1:size(bins,1)
    bin_size_sum = bin_size_sum + size(bins{i,1},1);
end
fprintf('Average bin size: %d\n',bin_size_sum / size(bins,1));

% Train classifiers
nodeIndices = kdGetAllNodeIndices(tree);

dataNodes = zeros(size(data,1),1);
for i = 1:size(dataNodes,1)
    dataNodes(i) = kdGetBinIndex(tree,data_iso(i,:));
end
%%

%% Define datasets
datasetName = {};
fileName = {};
classlabel = {};

datasetName{end+1} = 'Crowdsource';
fileName{end+1} = 'crs';
classlabel{end+1} = (results_voting(plotted_indices) >= crowdsourceCutoff);

datasetName{end+1} = 'Physical testing';
fileName{end+1} = 'phy';
classlabel{end+1} = (groundtruth >= physicalCutoff);

datasetName{end+1} = 'Energy';
fileName{end+1} = 'egy';
classlabel{end+1} = (results1(plotted_indices) <= energycutoff);

%%

if(~filter_with_confidence)
    confidence_thresholds = 0;
    binThreshold = size(data,1)+1;
else
    %confidence_thresholds = .9;
    confidence_thresholds = [0 .5:.1:.9];
    %confidence_thresholds = [.5 .8];
end

for confidence_cutoff = confidence_thresholds
    
    meanfunc = @meanConst;
    covfunc = @covSEard;
    likfunc = @likGauss;
    inffunc = @infExact;

    xdepth = finalDimensionNum; % # of dimensions
    covdepth = xdepth+1;
    
    %auc = zeros(1,loopn);
    fprCutoffs = [.05, .10, .15];
    
    predictions = cell(loopn,length(fprCutoffs),length(classlabel));
    trueValues = cell(loopn,length(fprCutoffs),length(classlabel));
    trueValues_unfiltered = cell(loopn,length(fprCutoffs),length(classlabel));
    prediction_points = zeros(loopn,length(fprCutoffs),length(classlabel));
    
    auc = cell(length(classlabel),length(fprCutoffs));
    rocx = cell(length(classlabel),length(fprCutoffs));
    rocy = cell(length(classlabel),length(fprCutoffs));
    
    % Passed to contour3d - added to each iteration
    mfout = [];
    mfout_precision = [];
    mfout_confidence = [];
    mfout_TP = [];
    indexout = [];
    indexout_precision = [];
    indexout_confidence = [];
    indexout_TP = [];
    
    % Remember precisions for every bin over all runs
    bin_precisions = zeros(loopn,size(bins,1));
    
    isomap_testIndices = cell(length(classlabel),length(fprCutoffs));
    isomap_predictions = cell(length(classlabel),length(fprCutoffs));
    
    aves = cell(length(classlabel),length(fprCutoffs));
    uppererrs = cell(length(classlabel),length(fprCutoffs));
    lowererrs = cell(length(classlabel),length(fprCutoffs));
    
    hyp_dump = cell(loopn,length(classlabel));
    
    l = 1;
    while(l <= loopn)
        hyps = {};
        for s = 1:length(classlabel)
            hyp = [];
            hyp.mean = zeros(1,1);
            hyp.cov(1:covdepth) = log(1);
            hyp.lik(1:1) = -.2;
            hyps{end+1} = hyp;
        end
        
        % Make testx vector of appropriate size
        [validationIndices, testIndices, trainIndices] = partitionData([validationProportion; testProportion],size(data_pca,1));
        
        for s = 1:length(classlabel)
            hyps{s} = minimize(hyps{s}, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, data_pca(trainIndices,:), classlabel{s}(trainIndices));
            hyp_dump{l,s} = hyps{s};
        end
        
        % Keeps track of which predictions were from which nodes
        test_containing_nodes = zeros(size(testIndices,1),1);
        % We want these for all test indices including the eliminated ones
        for i = 1:size(test_containing_nodes,1)
            idx = kdGetBinIndex(tree,data_iso(testIndices(i),:));
            test_containing_nodes(i) = idx;
        end
        
        % Classify the train2 points
        mf = {};
        for s = 1:length(classlabel)
            [mf{s}, ~, ~, ~] = gp(hyps{s}, inffunc, meanfunc, covfunc, likfunc,data_pca(trainIndices,:),classlabel{s}(trainIndices),data_pca(validationIndices,:));
            lrCutoff{s} = zeros(length(fprCutoffs),1);
            for k = 1:length(fprCutoffs)
                lrCutoff{s}(k) = fprThreshold(+(groundtruth(validationIndices) >= physicalCutoff),mf{s},fprCutoffs(k));
                fprintf('Using %f threshold of %f for classifier %s\n',fprCutoffs(k),lrCutoff{s}(k),datasetName{s});
            end
        end
        
        
        % As of 7/14/14, we only need 1 GP's confidence values. This
        % "primary" GP decides which points everyone predicts on for the head-to-head.
        confidences = zeros(length(nodeIndices),length(fprCutoffs));
        skipIteration = false;
        for k = 1:length(fprCutoffs)
            %confidences(:,k) = getNodeAccuracies(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff(k), groundtruth(validationIndices) >= physicalCutoff);
            confidences(:,k) = getNodeF1Scores(dataNodes(validationIndices), nodeIndices, mf{1} >= lrCutoff{1}(k), groundtruth(validationIndices) >= physicalCutoff);
            %confidences(:,k) = getNodeCorrelations(dataNodes(validationIndices), nodeIndices, mf{1}, groundtruth(validationIndices) >= physicalCutoff);

            % For every point in test set:
            % Find kdtree index
            % If confidence for the index is not high, remove from test set
            [~, ~, eliminateIndices(:,k)] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences(:,k), tree, data_iso(testIndices,:));

            fprintf('Eliminated %d points for FPR=%f\n',sum(eliminateIndices(:,k)),fprCutoffs(k));
            testIndices_final{k} = testIndices(~eliminateIndices(:,k));

            if size(data_pca(testIndices_final{k},:),1) < 10
                skipIteration = true;
                break;
            end
        end
        
        if skipIteration
            disp('Not enough valid test points; skipping iteration.');
            continue;
        end
        
        % Keeps track of which predictions were from which nodes
        test_containing_nodes = zeros(size(testIndices,1),1);
        % We want these for all test indices including the eliminated ones
        for i = 1:size(test_containing_nodes,1)
            idx = kdGetBinIndex(tree,data_iso(testIndices(i),:));
            test_containing_nodes(i) = idx;
        end
        
        for s = 1:length(classlabel)
            % Classify the filtered points
            [mf_unfiltered{s}, ~, ~, ~] = gp(hyps{s}, inffunc, meanfunc, covfunc, likfunc,data_pca(trainIndices,:),classlabel{s}(trainIndices),data_pca(testIndices,:));
        end
       
        res_unfiltered = +(groundtruth(testIndices) >= physicalCutoff);
        
        for k = 1:length(fprCutoffs)
            if sum(res_unfiltered(~eliminateIndices(:,k)) == 0) == 0 || sum(res_unfiltered(~eliminateIndices(:,k)) == 1) == 0
                skipIteration = true;
                break;
            end
        end
        if skipIteration
            disp('Filtered down to degenerate test set; skipping iteration.');
            continue;
        end
        
        for s = 1:length(classlabel)
            for k = 1:length(fprCutoffs)
                res = +(groundtruth(testIndices_final{k}) >= physicalCutoff);
                mf = mf_unfiltered{s}(~eliminateIndices(:,k));

                % Generate ROC curve
                try
                    mf_ufinal = mf_unfiltered{s};
                    % normalize only remaining points to massage the perfcurve
                    mf_ufinal(~eliminateIndices(:,k)) = normalizeer(mf_ufinal(~eliminateIndices(:,k)));
                    mf_ufinal(eliminateIndices(:,k)) = NaN;
                    [perfx,perfy,perft,auc{s,k}(end+1)] = perfcurve(res_unfiltered',mf_ufinal',1,'ProcessNaN','ignore');
                    %rocResults = [rocResults; [perfx perfy]];
                    [rocx{s,k}(:,end+1),rocy{s,k}(:,end+1)] = rotandproject(perfx,perfy,.01);
                catch err
                    auc(l) = NaN;
                end


                isomap_predictions{s,k} = [isomap_predictions{s,k}; (mf >= lrCutoff{s}(k))];
                isomap_testIndices{s,k} = [isomap_testIndices{s,k}; testIndices_final{k}];
                predictions{l,k,s} = (mf >= lrCutoff{s}(k));
                trueValues{l,k,s} = res;
                trueValues_unfiltered{l,k,s} = res_unfiltered;
                prediction_points(l,k,s) = length(testIndices_final{k});
            end
        end
        
%         node_precisions = zeros(size(nodeIndices));
%         % Generate the precision of all bins/nodes
%         for i = 1:size(nodeIndices,1)
%             % This recall value is not accurate. Update the allTestPoints
%             % value if recall is needed in the future
%             [node_precisions(i), ~] = precisionAndRecall({mf(test_containing_nodes_filtered==nodeIndices(i)) >= lrCutoff},{res(test_containing_nodes_filtered==nodeIndices(i))},true,{res(test_containing_nodes_filtered==nodeIndices(i))});
%         end
%         bin_precisions(l,:) = node_precisions;
        
        fprintf('Completed iteration %d\n',l);
        l = l+1;
    end
    
    %if confidence_cutoff == 0
        %% create isomap visualizations
        
        flipX = [0 1 0; 1 0 0; 0 0 0];
        flipY = [0 0 1; 1 0 1; 0 1 0];
        
        strongPts = cell(length(classlabel),length(fprCutoffs));
        iso_accuracies = cell(length(classlabel),length(fprCutoffs));
        
        for s = 1:length(classlabel)
            % TODO: the cutoff for the predictions going into isomap is based on
            % validation, not post-hoc testing. If we really want to know the
            % accuracies for 5/10/15% FPR, the code needs to be modified.
            for i = 1:length(fprCutoffs)
                filterLevel = fprCutoffs(i);
                %% TODO: filter points for high confidence levels properly
                threshold = fprThreshold(res_unfiltered,mf_unfiltered{s},filterLevel);
                for k = 6:6
                    saveTitle = sprintf('%s/isomap_dumps/522_8-9-14/%s_pca_fpr%02d_conf_%d',savedir,fileName{s},filterLevel*100,confidence_cutoff*10);
                    graphTitle = sprintf('PCA plot for %s (FPR = %f) filtering=%.2f\nBlue/Red = correct/incorrect prediction',datasetName{s},filterLevel,confidence_cutoff);
                    plotAveragePredictions(data_pca,isomap_predictions{s,i},isomap_testIndices{s,i},groundtruth >= physicalCutoff,graphTitle,saveTitle);
                    saveTitle = sprintf('%s/isomap_dumps/522_8-9-14/%s_fpr%02d_k%d_conf_%d',savedir,fileName{s},filterLevel*100,k,confidence_cutoff*10);
                    graphTitle = sprintf('Visualization for %s (FPR = %f, k = %d) filtering=%.2f\nBlue/Red = correct/incorrect prediction',datasetName{s},filterLevel,k,confidence_cutoff);
                    if s==1 && i==1
                        [visualizedPoints,iso_x,iso_y,iso_accuracies{s,i}] = isomapAveragePredictions(data,isomap_predictions{s,i},isomap_testIndices{s,i},groundtruth >= physicalCutoff,graphTitle,k,saveTitle,flipX(s,i),flipY(s,i));
                    else
                        [visualizedPoints,~,~,iso_accuracies{s,i}] = isomapAveragePredictions(data,isomap_predictions{s,i},isomap_testIndices{s,i},groundtruth >= physicalCutoff,graphTitle,k,saveTitle,flipX(s,i),flipY(s,i));
                    end
                    fprintf('Displayed %d points\n',visualizedPoints);
                    strongPts{s,i} = visualizedPoints(iso_accuracies{s,i} >= .5);
                end
            end
        end
        
        %%
    %end
    
    fprintf('\n\n');
    for k = 1:length(fprCutoffs)
        % Intersect crowdsource and physical testing
        crowdPts = setdiff(strongPts{1,k},strongPts{2,k});
        physicalPts = setdiff(strongPts{2,k},strongPts{1,k});
        
        fprintf('Strong crowdsource points for fprCutoff %f:\n',fprCutoffs(k));
        disp('Index,x,y,accuracy,groundtruth');
        for i = 1:length(crowdPts)
            idx = find(visualizedPoints == crowdPts(i));
            fprintf('%d,%f,%f,%f,%.1f\n',crowdPts(i),iso_x(idx),iso_y(idx),iso_accuracies{1,k}(idx),groundtruth(crowdPts(i)));
        end
        disp('Strong physical testing points:');
        disp('Index,x,y,accuracy,groundtruth');
        for i = 1:length(physicalPts)
            idx = find(visualizedPoints == physicalPts(i));
            fprintf('%d,%f,%f,%f,%.1f\n',physicalPts(i),iso_x(idx),iso_y(idx),iso_accuracies{2,k}(idx),groundtruth(physicalPts(i)));
        end
    end
    fprintf('\n\n');

    for s = 1:length(classlabel)
        for k = 1:length(fprCutoffs)
            fprintf('Overall results, FPR=%f, classifier %s\n',fprCutoffs(k),datasetName{s});
            if(filter_with_confidence)
                disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
            end

            [aves{s,k},uppererrs{s,k},lowererrs{s,k},upperstd,~,~,~,~] = rotandextrap(rocx{s,k},rocy{s,k},loopn);

            [ tpr,tprerr,tprstd ] = tprextfun( aves{s,k},upperstd,uppererrs{s,k},fpr );
            disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
            disp([fpr;tpr;tprerr;tprstd])

            %% Dump all high-precision grasp nodes
    %         fprintf('Printing all grasps falling into high-precision nodes. Conf=%f\n',confidence_cutoff);
    %         pointInfos = [(1:size(data_pca,1))' data_pca];
    %         totalPointsDumped = 0;
    %         for i = 1:size(nodeIndices,1)
    %             if isnan(averageBinPrecisions(i)) || averageBinPrecisions(i) < .9
    %                 continue;
    %             end
    %             nodeDump = pointInfos(dataNodes == nodeIndices(i),:);
    %             totalPointsDumped = totalPointsDumped+size(nodeDump,1);
    %             fprintf('Node index %d\n',nodeIndices(i));
    %             disp(nodeDump);
    %         end
    %         fprintf('%d points dumped in total\n',totalPointsDumped);
    %         clearvars pointInfos nodeDump totalPointsDumped
    %         point_precisions = zeros(size(data_pca,1),1);
    %         for i = 1:length(point_precisions)
    %             point_precisions(i) = averageBinPrecisions(dataNodes(i) == nodeIndices);
    %             if point_precisions(i) > .90
    %                 fprintf('index %d: %f, %f\n',i,data_pca(i,1),data_pca(i,2));
    %             end
    %         end
            %%

            fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points(:,k,s)),std(prediction_points(:,k,s)));

            %Display AUC Info
            disp('Individual AUCs:');
            auc_ = auc{s,k};
            auc_ = auc_(~isnan(auc_)); % Clear any degenerate AUC points
            disp(auc_');
            fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc_));
            disp('AUC ave err std 95% conf')
            disp(mean(auc_))
            disp(std(auc_)/(length(auc_)^.5))
            disp(std(auc_))
            disp(1.96*std(auc_)/(length(auc_)^.5));

            [precision_macro, recall_macro, p_err, r_err, recall_macro_loc, r_l_err] = precisionAndRecall(predictions(:,k,s),trueValues(:,k,s),true,trueValues_unfiltered(:,k,s));
            disp('Precision / Recall / Local Recall macro');
            disp([precision_macro recall_macro recall_macro_loc]);
            disp('');
            [precision_micro, recall_micro, p_err, r_err, recall_micro_loc, r_l_err] = precisionAndRecall(predictions(:,k,s),trueValues(:,k,s),false,trueValues_unfiltered(:,k,s));
            disp('Precision / Recall micro');
            disp([precision_micro recall_micro recall_micro_loc]);

            disp('Std devs');
            disp([p_err r_err r_l_err]);
            
            disp('F1 score (based on micro averaging)');
            disp(2*precision_micro*recall_micro_loc/(precision_micro+recall_micro_loc));
        end
    end
    
    generateROC(sprintf('%s/rocdumps/roc_conf_%d',savedir,confidence_cutoff*10),uppererrs(:,1),lowererrs(:,1),aves(:,1),datasetName,sprintf('ROC at confidence %.2f',confidence_cutoff));
end

% disp('    TPR       FPR       TNR       FNR       cutoff');
% disp(summaryStats);

warning(orig_warn_state);

thesisCleanup;
end
