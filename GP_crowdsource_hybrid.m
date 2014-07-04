%% GP_precision_baseline.m
%% 6/30/14
%  Trains multiple GPs, one for each class label category, and attempts to
%  find the best classifier for different kd-tree nodes.
%%
function GP_crowdsource_hybrid(crowdsourceCutoff,energycutoff,binThreshold)

%me = mfilename;
me = sprintf('GP_crs_%d_egy_%d_bin_%d',crowdsourceCutoff*100,energycutoff,binThreshold);
use_timestamp = true;
thesisStartup;

% Hard-coded for printing convenience
metricNames = 'PointArng TriSize Extension Spread Limit PerpSym ParallelSym OrthoNorm Volume GraspIt_Volume GraspIt_Epsilon';

% Suppress pesky LR warnings
orig_warn_state = warning;
warning('off','all');

% Number of features to read
dimensionCount = 11;

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

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

% Linear regression prediction must be >= this value to be positive
lrCutoff = .25;
fprCutoff = .25;

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 30;

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .25;
validationProportion = .25;
train2Proportion = .25;

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
tree = kdMedCreateTree(data,binThreshold);
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
    dataNodes(i) = kdGetBinIndex(tree,data(i,:));
end
%%

if(~filter_with_confidence)
    confidence_thresholds = 0;
    binThreshold = size(data,1)+1;
else
    confidence_thresholds = [0 .5:.1:.9];
    %confidence_thresholds = [.5 .8];
end
% Subtract/add a small number to prevent minimum value being placed in
% bin_index 0 and maximum in 11.
PCA_min = min(data)-.0001;
PCA_max = max(data)+.0002;
for confidence_cutoff = confidence_thresholds
    
    meanfunc = @meanConst;
    covfunc = @covSEard;
    likfunc = @likGauss;
    inffunc = @infExact;

    xdepth = finalDimensionNum; % # of dimensions
    covdepth = xdepth+1;
    
    %% Define class labels
    datasets = {'energy','crowdsource'};
    y_physical = (groundtruth >= physicalCutoff);
    y_energy = (results1 <= energycutoff);
    y_crowdsource = (results_voting >= crowdsourceCutoff);
    y_all = {y_energy, y_crowdsource};
    %energycutoff = crowdsourceCutoff;
    %%
    
    predictions_hybrid = {};
    trueValues_hybrid = {};
    predictions = cell(loopn,length(y_all));
    trueValues = cell(loopn,length(y_all));
    trueValues_unfiltered = cell(loopn,length(y_all));
    prediction_points = zeros(loopn,length(y_all));
    
    %auc = zeros(1,loopn);
    auc = zeros(loopn,length(y_all));
    rocx = cell(3,1);
    rocy = cell(3,1);
    %rocx = [];
    %rocy = [];
    
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
    
    l = 1;
    while(l <= loopn)
        trainData = data_pca;

        hyps = {};
        for i = 1:length(y_all)
            hyp = [];
            hyp.mean = zeros(1,1);
            hyp.cov(1:covdepth) = log(1);
            hyp.lik(1:1) = -.2;
            hyps{end+1} = hyp;
        end
        
        % Split data into partitions
        [train2Indices, validationIndices, testIndices, trainIndices] = partitionData([train2Proportion; validationProportion; testProportion],size(data_pca,1));

        %lrCutoff = zeros(1,length(y_all));
        confidences = zeros(length(nodeIndices),length(y_all));
        
        % Used for switching nodes to different classifiers in hybrid
        filterFPRs = [.05 .10 .15];
        trainThresholds = zeros(length(filterFPRs),length(y_all));
        hybridConfidences = cell(length(filterFPRs),1);
%         for i = 1:length(filterFPRs)
%             hybridConfidences{end+1} = zeros(length(nodeIndices),length(y_all));
%         end
        
        for i = 1:length(y_all)
            hyps{i} = minimize(hyps{i}, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainData(trainIndices,:), y_all{i}(trainIndices));
            % Classify the train2 points
            [mf, ~, ~, ~] = gp(hyps{i}, inffunc, meanfunc, covfunc, likfunc,trainData(train2Indices,:),y_all{i}(train2Indices),trainData(validationIndices,:));
            %lrCutoff(i) = fprThreshold(+(groundtruth(validationIndices) >= physicalCutoff),mf,fprCutoff);
            %fprintf('Using threshold of %f\n',lrCutoff(i));
            
            for filterLevel = 1:length(filterFPRs);
                trainThresholds(filterLevel,i) = fprThreshold(+(groundtruth(validationIndices) >= physicalCutoff),mf,filterFPRs(filterLevel));
                hybridConfidences{filterLevel}(:,i) = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= trainThresholds(filterLevel,i), groundtruth(validationIndices) >= physicalCutoff);
            end

            % For every index in nodeIndices:
            % compute the confidence rating by looking at ground truth
            % create logical vector of same size as nodeIndices, saying whether
            % or not to predict in bin.
            %confidences(:,i) = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff(i), groundtruth(validationIndices) >= physicalCutoff);
            confidences(:,i) = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff, groundtruth(validationIndices) >= physicalCutoff);
        end
        
        

        % For every point in test set:
        % Find kdtree index
        % If confidence for the index is not high, remove from test set
        [testx, ~, ~] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences(:,1), tree, data(testIndices,:));
        
        elimIndices_hybrid = {};
        for i = 1:length(y_all)
            %[~,~,elimIndices_hybrid{end+1}] = filterPredictionSet(confidence_cutoff, nodeIndices, hybridConfidences{3}(:,i), tree, data(testIndices,:));
            [~,~,elimIndices_hybrid{end+1}] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences(:,i), tree, data(testIndices,:));
        end
        %[testx_hybrid, eliminated_hybrid, elimIndices_hybrid_final] = filterPredictionSet(confidence_cutoff, nodeIndices, max(hybridConfidences{3},[],2), tree, data(testIndices,:));
        [testx_hybrid, eliminated_hybrid, elimIndices_hybrid_final] = filterPredictionSet(confidence_cutoff, nodeIndices, max(confidences,[],2), tree, data(testIndices,:));
        
        fprintf('Eliminated %d points\n',sum(elimIndices_hybrid_final));
        
        if size(testx,1) < 10
            disp('Not enough valid test points; skipping iteration.');
            continue;
        end
        
        % Keeps track of which predictions were from which nodes
        test_containing_nodes = zeros(size(testIndices,1),1);
        % We want these for all test indices including the eliminated ones
        for i = 1:size(test_containing_nodes,1)
            idx = kdGetBinIndex(tree,data(testIndices(i),:));
            test_containing_nodes(i) = idx;
        end
        
        % Classify the filtered points
        %[mf_unfiltered, ~, ~, ~] = gp(hyps{1}, inffunc, meanfunc, covfunc, likfunc,data_pca(train2Indices,:),y_all{1}(train2Indices),data_pca(testIndices,:));
        
        
        %% Get hybrid predictions
        res_unfiltered = +(groundtruth(testIndices) >= cutoff);
        
        mf_single = zeros(length(testIndices),length(y_all));
        %[~,bestClassifier] = max(hybridConfidences{3},[],2);
        [~,bestClassifier] = max(confidences,[],2);
        for i = 1:length(y_all)
            %res = +(groundtruth(testIndices_final) >= cutoff);
            testIndices_hybrid = testIndices(~elimIndices_hybrid{i});
            res = +(groundtruth(testIndices_hybrid) >= cutoff);

            %mf = mf_unfiltered(~elimIndices_hybrid);
            %mf = mf_hybrid_final(~elimIndices_hybrid);
            %mf_unfiltered = normalizeer(mf_hybrid_final);
        
            [mf_single(:,i), ~, ~, ~] = gp(hyps{i}, inffunc, meanfunc, covfunc, likfunc,data_pca(train2Indices,:),y_all{i}(train2Indices),data_pca(testIndices,:));
            % Just using a default .5 threshold for now
            mf_hybrid_filtered = mf_single(~elimIndices_hybrid{i},i);
            %predictions{l,i} = (mf_hybrid_filtered >= trainThresholds(3,i));
            predictions{l,i} = (mf_hybrid_filtered >= lrCutoff);
            trueValues{l,i} = res;
            trueValues_unfiltered{l,i} = res_unfiltered;
            prediction_points(l,i) = length(testIndices(~elimIndices_hybrid{i}));
        end
        
        % Do the same for the hybrid
        testIndices_final_hybrid = testIndices(~elimIndices_hybrid_final);
        res = +(groundtruth(testIndices_final_hybrid) >= cutoff);
        mf_hybrid_final = zeros(length(testIndices),1);
        for n = 1:size(testIndices,1)
            idx = test_containing_nodes(n)==nodeIndices;
            mf_hybrid_final(n) = mf_single(n,bestClassifier(idx));
            %mf_hybrid_final(n) = mf_single(n,1);
        end
        mf_hybrid_final_filtered = mf_hybrid_final(~elimIndices_hybrid_final);
        %TODO: predictions need to be thresholded on the appropriate classifier
        %predictions_hybrid{l} = (mf_hybrid_final_filtered >= trainThresholds(3,i));
        predictions_hybrid{l} = (mf_hybrid_final_filtered >= lrCutoff);
        trueValues_hybrid{l} = res;
        prediction_points_hybrid(l) = length(testIndices(~elimIndices_hybrid_final));
        %%
        
        %% Generate ROC curve
        for k = 1:length(y_all)
            try
                mf_single(elimIndices_hybrid{k},k) = NaN;
                [perfx,perfy,perft,auc(l,k)] = perfcurve(res_unfiltered',mf_single(:,k)',1,'ProcessNaN','ignore');
                %rocResults = [rocResults; [perfx perfy]];
                [rocx{k}(:,end+1),rocy{k}(:,end+1)] = rotandproject(perfx,perfy,.01);
            catch err
                auc(l,k) = NaN;
            end
        end
        % final hybrid
        try
            mf_hybrid_final(elimIndices_hybrid_final) = NaN;
            [perfx,perfy,perft,auc_hybrid(l)] = perfcurve(res_unfiltered',mf_hybrid_final',1,'ProcessNaN','ignore');
            %rocResults = [rocResults; [perfx perfy]];
            [rocx{end}(:,end+1),rocy{end}(:,end+1)] = rotandproject(perfx,perfy,.01);
        catch err
            auc_hybrid(l) = NaN;
        end
        %%
        
        %% TODO: update for multiple classifiers
%         node_precisions = zeros(size(nodeIndices));
%         % Generate the precision of all bins/nodes
%         for i = 1:size(nodeIndices,1)
%             % This recall value is not accurate. Update the allTestPoints
%             % value if recall is needed in the future
%             [node_precisions(i), ~] = precisionAndRecall({mf(test_containing_nodes_filtered_hybrid==nodeIndices(i)) >= lrCutoff(1)},{res(test_containing_nodes_filtered_hybrid==nodeIndices(i))},true,{res(test_containing_nodes_filtered_hybrid==nodeIndices(i))});
%         end
%         bin_precisions(l,:) = node_precisions;
        
        fprintf('Completed iteration %d\n',l);
        l = l+1;
    end
    
    %% Generate overall-precision contour
    contour_file = strcat(savedir,'hybrid_contours\',sprintf('GP_crs_%d_conf_%d_bin_%d',crowdsourceCutoff*100,confidence_cutoff*10,binThreshold));
    averageBinPrecisions = mean(bin_precisions)';
    
    %% Dump all high-precision grasp nodes
%     fprintf('Printing all grasps falling into high-precision nodes. Conf=%f\n',confidence_cutoff);
%     pointInfos = [(1:size(data_pca,1))' data_pca];
%     totalPointsDumped = 0;
%     for i = 1:size(nodeIndices,1)
%         if isnan(averageBinPrecisions(i)) || averageBinPrecisions(i) < .9
%             continue;
%         end
%         nodeDump = pointInfos(dataNodes == nodeIndices(i),:);
%         totalPointsDumped = totalPointsDumped+size(nodeDump,1);
%         fprintf('Node index %d\n',nodeIndices(i));
%         disp(nodeDump);
%     end
%     fprintf('%d points dumped in total\n',totalPointsDumped);
%     clearvars pointInfos nodeDump totalPointsDumped
%     point_precisions = zeros(size(data_pca,1),1);
%     for i = 1:length(point_precisions)
%         point_precisions(i) = averageBinPrecisions(dataNodes(i) == nodeIndices);
%         if point_precisions(i) > .90
%             fprintf('index %d: %f, %f\n',i,data_pca(i,1),data_pca(i,2));
%         end
%     end
    %%
    
    %generateCategorizedContour(data_pca,groundtruth,(1:length(point_precisions))',point_precisions,contour_file,sprintf('Areas of high test precision\nGlobal GP, confidence cutoff=%f',confidence_cutoff),[.5 .75 .9]);
    %save(sprintf('%scontourdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*100),'data_pca','groundtruth','point_precisions','contour_file','confidence_cutoff');
    
    %%
    
        %% Generate AUC curve
    %save(sprintf('%srocdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*100),'uppererr','lowererr','ave');
%     fig = figure;
%     set(fig, 'units', 'inches', 'pos', [8 5 3.25 3])
%     set(gca,'FontSize',10)
%     fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
%     %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
%     hold on
%     plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
%     %plot(rocResults(:,1).*100,rocResults(:,2).*100,'k','linewidth',2)
%     plot([0 100],[0 100],'color',[.5 .5 .5])
%     set(gca,'box','on','position',[.17,.15,.78,.8])
%     axis square
%     axis([0 100 0 100])
%     ylabel({'TPR';'(%)'})
%     xlabel('FPR (%)')
%     set(get(gca,'YLabel'),'Rotation',0)
%     plot([40 45],[52,60],'k','linewidth',1)
%     set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
%     text(10,84,'GP with PC1','FontSize',10)
%     text(10,8,'Random Guess','FontSize',10)
    %%
    
    if(filter_with_confidence)
        disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
    end
    for k = 1:length(y_all)
        fprintf('Overall results (%s)\n',datasets{k});
        
        [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx{k},rocy{k},size(auc,1));

        [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
        disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
        disp([fpr;tpr;tprerr;tprstd])

        fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points(:,k)),std(prediction_points(:,k)));

        %Display AUC Info
    %    disp('Individual AUCs:');
        auc_ = auc(:,k);
        auc_ = auc_(~isnan(auc_)); % Clear any degenerate AUC points
    %    disp(auc');
        fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc_));
        disp('AUC ave err std')
        disp(mean(auc_))
        disp(std(auc_)/(size(auc_,1))^.5)
        disp(std(auc_))

        [precision_macro, recall_macro, p_err, r_err] = precisionAndRecall(predictions(:,k),trueValues(:,k),true,trueValues_unfiltered(:,k));
        disp('Precision / Recall macro');
        disp([precision_macro recall_macro]);
        disp('');
        [precision_micro, recall_micro] = precisionAndRecall(predictions(:,k),trueValues(:,k),false,trueValues_unfiltered(:,k));
        disp('Precision / Recall micro');
        disp([precision_micro recall_micro]);

        disp('Std devs');
        disp([p_err r_err]);
    end
    
    disp('Overall results (hybrid)');
    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx{end},rocy{end},size(auc_hybrid,1));

    [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
    disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
    disp([fpr;tpr;tprerr;tprstd])

    fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points_hybrid),std(prediction_points_hybrid));

    %Display AUC Info
%    disp('Individual AUCs:');
    auc_ = auc_hybrid;
    auc_ = auc_(~isnan(auc_)); % Clear any degenerate AUC points
%    disp(auc');
    fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc_));
    disp('AUC ave err std')
    disp(mean(auc_))
    disp(std(auc_)/(size(auc_,1))^.5)
    disp(std(auc_))

    [precision_macro, recall_macro, p_err, r_err] = precisionAndRecall(predictions_hybrid,trueValues_hybrid,true,trueValues_unfiltered(:,end));
    disp('Precision / Recall macro');
    disp([precision_macro recall_macro]);
    disp('');
    [precision_micro, recall_micro] = precisionAndRecall(predictions_hybrid,trueValues_hybrid,false,trueValues_unfiltered(:,end));
    disp('Precision / Recall micro');
    disp([precision_micro recall_micro]);

    disp('Std devs');
    disp([p_err r_err]);
end

% disp('    TPR       FPR       TNR       FNR       cutoff');
% disp(summaryStats);

warning(orig_warn_state);

thesisCleanup;
end
