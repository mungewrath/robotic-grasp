%% GP_precision_baseline.m
%% 6/20/14
%  Trains a GP and outputs predictions based on crowdsource. Used for
%  the poster as a baseline for the LR hybrid classifier.
%%
function GP_crowdsource(crowdsourceCutoff,binThreshold)

%me = mfilename;
me = sprintf('GP_phy_%d_bin_%d',crowdsourceCutoff*100,binThreshold);
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
lrCutoff = .5;
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
    %confidence_thresholds = 0;
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
    
    predictions = {};
    trueValues = {};
    trueValues_unfiltered = {};
    prediction_points = zeros(loopn,1);
    
    %auc = zeros(1,loopn);
    auc = [];
    rocx = [];
    rocy = [];
    
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
        hyp = [];
        hyp.mean = zeros(1,1);
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:1) = -.2;
    
        datasetName = 'Physical testing';
        y = (groundtruth >= physicalCutoff);
        %datasetName = 'Crowdsource';
        %y = (results_voting >= crowdsourceCutoff);
        %datasetName = 'Energy';
        energycutoff = crowdsourceCutoff;
        %y = (results1 <= energycutoff);
        
        % Make testx vector of appropriate size
        [train2Indices, validationIndices, testIndices, trainIndices] = partitionData([train2Proportion; validationProportion; testProportion],size(data_pca,1));
        
        hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, data_pca(trainIndices,:), y(trainIndices));
        
        % Keeps track of which predictions were from which nodes
        test_containing_nodes = zeros(size(testIndices,1),1);
        % We want these for all test indices including the eliminated ones
        for i = 1:size(test_containing_nodes,1)
            idx = kdGetBinIndex(tree,data(testIndices(i),:));
            test_containing_nodes(i) = idx;
        end
        
        % Classify the train2 points
        %[mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,train2x,train2y,validationx);
        [mf, ~, ~, ~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,data_pca(train2Indices,:),y(train2Indices),data_pca(validationIndices,:));
        %lrCutoff = fprThreshold(+(groundtruth(validationIndices) >= physicalCutoff),mf,fprCutoff);
        %fprintf('Using threshold of %f\n',lrCutoff);
        
        confidences = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff, groundtruth(validationIndices) >= physicalCutoff);
        confidences = getNodeAccuracies(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff, groundtruth(validationIndices) >= physicalCutoff);

        % For every point in test set:
        % Find kdtree index
        % If confidence for the index is not high, remove from test set
        [~, ~, eliminateIndices] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences(:,1), tree, data(testIndices,:));
        
        fprintf('Eliminated %d points\n',sum(eliminateIndices));
        testIndices_final = testIndices(~eliminateIndices);
        
        if size(data_pca(testIndices_final,:),1) < 10
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
        test_containing_nodes_filtered = test_containing_nodes(~eliminateIndices);
        
        % Classify the filtered points
        [mf_unfiltered, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,data_pca(train2Indices,:),y(train2Indices),data_pca(testIndices,:));
       
        res = +(groundtruth(testIndices_final) >= cutoff);
        res_unfiltered = +(groundtruth(testIndices) >= cutoff);
        mf = mf_unfiltered(~eliminateIndices);
        mf_unfiltered = normalizeer(mf_unfiltered);
        
        %% create isomap visualizations 
%         filterFPRs = [.05, .10, .15];
%         for filterLevel = filterFPRs
%             threshold = fprThreshold(res_unfiltered,mf_unfiltered,filterLevel);
%             for k = 2:7
%                 visualizedPoints = isomapPredictions(data(testIndices,:),mf_unfiltered >= threshold,res_unfiltered,sprintf('Visualization for %s (FPR = %f, k = %d)\nBlue/Red = correct/incorrect prediction',datasetName,filterLevel,k),k);
%                 fprintf('Displayed %d points\n',visualizedPoints);
%                 fprintf('%d / %d points correct\n',sum((mf_unfiltered >= threshold) == res_unfiltered),length(res_unfiltered));
%             end
%         end
        %%
        
        % Generate ROC curve
        try
            mf_unfiltered(eliminateIndices) = NaN;
            [perfx,perfy,perft,auc(end+1)] = perfcurve(res_unfiltered',mf_unfiltered',1,'ProcessNaN','ignore');
            %rocResults = [rocResults; [perfx perfy]];
            [rocx(:,end+1),rocy(:,end+1)] = rotandproject(perfx,perfy,.01);
        catch err
            auc(l) = NaN;
        end
        
        
        predictions{l} = (mf >= lrCutoff);
        trueValues{l} = res;
        trueValues_unfiltered{l} = res_unfiltered;
        prediction_points(l) = length(testIndices_final);
        
        node_precisions = zeros(size(nodeIndices));
        % Generate the precision of all bins/nodes
        for i = 1:size(nodeIndices,1)
            % This recall value is not accurate. Update the allTestPoints
            % value if recall is needed in the future
            [node_precisions(i), ~] = precisionAndRecall({mf(test_containing_nodes_filtered==nodeIndices(i)) >= lrCutoff},{res(test_containing_nodes_filtered==nodeIndices(i))},true,{res(test_containing_nodes_filtered==nodeIndices(i))});
        end
        bin_precisions(l,:) = node_precisions;
        
        fprintf('Completed iteration %d\n',l);
        l = l+1;
    end
    
    %% Generate overall-precision contour
    contour_file = strcat(savedir,'hybrid_contours\',sprintf('GP_crs_%d_conf_%d_bin_%d',crowdsourceCutoff*100,confidence_cutoff*10,binThreshold));
    averageBinPrecisions = mean(bin_precisions)';
    
    %% Dump all high-precision grasp nodes
    fprintf('Printing all grasps falling into high-precision nodes. Conf=%f\n',confidence_cutoff);
    pointInfos = [(1:size(data_pca,1))' data_pca];
    totalPointsDumped = 0;
    for i = 1:size(nodeIndices,1)
        if isnan(averageBinPrecisions(i)) || averageBinPrecisions(i) < .9
            continue;
        end
        nodeDump = pointInfos(dataNodes == nodeIndices(i),:);
        totalPointsDumped = totalPointsDumped+size(nodeDump,1);
        fprintf('Node index %d\n',nodeIndices(i));
        disp(nodeDump);
    end
    fprintf('%d points dumped in total\n',totalPointsDumped);
    clearvars pointInfos nodeDump totalPointsDumped
    point_precisions = zeros(size(data_pca,1),1);
    for i = 1:length(point_precisions)
        point_precisions(i) = averageBinPrecisions(dataNodes(i) == nodeIndices);
        if point_precisions(i) > .90
            fprintf('index %d: %f, %f\n',i,data_pca(i,1),data_pca(i,2));
        end
    end
    %%
    
    %generateCategorizedContour(data_pca,groundtruth,(1:length(point_precisions))',point_precisions,contour_file,sprintf('Areas of high test precision\nGlobal GP, confidence cutoff=%f',confidence_cutoff),[.5 .75 .9]);
    save(sprintf('%scontourdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*100),'data_pca','groundtruth','point_precisions','contour_file','confidence_cutoff');
    
    %%
    
    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,size(auc,1));
    
    [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
    disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD')
    disp([fpr;tpr;tprerr;tprstd])
    
    %% Generate AUC curve
    save(sprintf('%srocdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*100),'uppererr','lowererr','ave');
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

    disp('Overall results (hybrid)');
    if(filter_with_confidence)
        disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
    end
    if(~filter_with_confidence)
        disp('All datapoints used');
    else
        disp('Filtered datapoints');
    end
    fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points),std(prediction_points));
    
    %Display AUC Info
    disp('Individual AUCs:');
    auc = auc(~isnan(auc)); % Clear any degenerate AUC points
    disp(auc');
    fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc));
    disp('AUC ave err std')
    disp(mean(auc))
    disp(std(auc)/(size(auc,1))^.5)
    disp(std(auc))
    
    [precision_macro, recall_macro, p_err, r_err] = precisionAndRecall(predictions,trueValues,true,trueValues_unfiltered);
    disp('Precision / Recall macro');
    disp([precision_macro recall_macro]);
    disp('');
    [precision_micro, recall_micro] = precisionAndRecall(predictions,trueValues,false,trueValues_unfiltered);
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
