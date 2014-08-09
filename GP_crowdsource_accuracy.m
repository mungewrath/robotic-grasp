%% GP_precision_baseline.m
%% 7/3/14
%  Trains a GP and outputs predictions based on crowdsource. Used for
%  the poster as a baseline for the LR hybrid classifier.
%  Looks at classifier success at a number of thresholds, based on FPR
%  cutoffs defined during validation.
%  Also replaces the usual confidence filtering with accuracy (correct /
%  total predictions)
%%
function GP_crowdsource_accuracy(crowdsourceCutoff,binThreshold)

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
    fprCutoffs = [.05, .10, .15];
    auc = cell(1,length(fprCutoffs));
    rocx = cell(1,length(fprCutoffs));
    rocy = cell(1,length(fprCutoffs));
    
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
    
    isomap_testIndices = cell(1,length(fprCutoffs));
    isomap_predictions = cell(1,length(fprCutoffs));
    
    hyps = {};
    
    l = 1;
    while(l <= loopn)
        hyp = [];
        hyp.mean = zeros(1,1);
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:1) = -.2;
    
        %datasetName = 'Physical testing';
        %fileName = 'phy';
        %y = (groundtruth >= physicalCutoff);
        datasetName = 'Crowdsource';
        fileName = 'crs';
        y = (results_voting >= crowdsourceCutoff);
        %datasetName = 'Energy';
        %fileName = 'egy';
        %energycutoff = crowdsourceCutoff;
        %y = (results1 <= energycutoff);
        
        % Make testx vector of appropriate size
        [train2Indices, validationIndices, testIndices, trainIndices] = partitionData([train2Proportion; validationProportion; testProportion],size(data_pca,1));
        
        hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, data_pca(trainIndices,:), y(trainIndices));
        hyps{l} = hyp;
        
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
        lrCutoff = zeros(length(fprCutoffs),1);
        for k = 1:length(fprCutoffs)
            lrCutoff(k) = fprThreshold(+(groundtruth(validationIndices) >= physicalCutoff),mf,fprCutoffs(k));
            fprintf('Using %f threshold of %f\n',fprCutoffs(k),lrCutoff(k));
        end
        
        
        %confidences = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff, groundtruth(validationIndices) >= physicalCutoff);
        confidences = zeros(length(nodeIndices),length(fprCutoffs));
        skipIteration = false;
        for k = 1:length(fprCutoffs)
            confidences(:,k) = getNodeAccuracies(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff(k), groundtruth(validationIndices) >= physicalCutoff);
            %confidences(:,k) = getNodeConfidences(dataNodes(validationIndices), nodeIndices, mf >= lrCutoff(k), groundtruth(validationIndices) >= physicalCutoff);

            % For every point in test set:
            % Find kdtree index
            % If confidence for the index is not high, remove from test set
            [~, ~, eliminateIndices(:,k)] = filterPredictionSet(confidence_cutoff, nodeIndices, confidences(:,k), tree, data(testIndices,:));

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
            idx = kdGetBinIndex(tree,data(testIndices(i),:));
            test_containing_nodes(i) = idx;
        end
        
        % Classify the filtered points
        [mf_unfiltered, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,data_pca(train2Indices,:),y(train2Indices),data_pca(testIndices,:));
       
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
        
        for k = 1:length(fprCutoffs)
            res = +(groundtruth(testIndices_final{k}) >= physicalCutoff);
            mf = mf_unfiltered(~eliminateIndices(:,k));

            % Generate ROC curve
            try
                mf_ufinal = mf_unfiltered;
                % normalize only remaining points to massage the perfcurve
                mf_ufinal(~eliminateIndices(:,k)) = normalizeer(mf_ufinal(~eliminateIndices(:,k)));
                mf_ufinal(eliminateIndices(:,k)) = NaN;
                [perfx,perfy,perft,auc{k}(end+1)] = perfcurve(res_unfiltered',mf_ufinal',1,'ProcessNaN','ignore');
                %rocResults = [rocResults; [perfx perfy]];
                [rocx{k}(:,end+1),rocy{k}(:,end+1)] = rotandproject(perfx,perfy,.01);
            catch err
                auc(l) = NaN;
            end


            isomap_predictions{k} = [isomap_predictions{k}; (mf >= lrCutoff(k))];
            isomap_testIndices{k} = [isomap_testIndices{k}; testIndices_final{k}];
            predictions{l,k} = (mf >= lrCutoff(k));
            trueValues{l,k} = res;
            trueValues_unfiltered{l,k} = res_unfiltered;
            prediction_points(l,k) = length(testIndices_final{k});
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
    
    if confidence_cutoff == 0
        %% create isomap visualizations
        
        % TODO: the cutoff for the predictions going into isomap is based on
        % validation, not post-hoc testing. If we really want to know the
        % accuracies for 5/10/15% FPR, the code needs to be modified.
        for i = 1:length(fprCutoffs)
            filterLevel = fprCutoffs(i);
            threshold = fprThreshold(res_unfiltered,mf_unfiltered,filterLevel);
            for k = 6:6
                saveTitle = sprintf('%s/isomap_dumps/522_7-14-14/%s_fpr%02d_k%d',savedir,fileName,filterLevel*100,k);
                graphTitle = sprintf('Visualization for %s (FPR = %f, k = %d)\nBlue/Red = correct/incorrect prediction',datasetName,filterLevel,k);
                visualizedPoints = isomapAveragePredictions(data,isomap_predictions{i},isomap_testIndices{i},groundtruth >= physicalCutoff,graphTitle,k,saveTitle,filterLevel==.1,filterLevel==.15);
                fprintf('Displayed %d points\n',visualizedPoints);
                fprintf('%d / %d points correct\n',sum((mf_unfiltered >= threshold) == res_unfiltered),length(res_unfiltered));
            end
        end
        %%
    end
    
    %% Generate overall-precision contour
    contour_file = strcat(savedir,'hybrid_contours\',sprintf('GP_crs_%d_conf_%d_bin_%d',crowdsourceCutoff*100,confidence_cutoff*10,binThreshold));
    averageBinPrecisions = mean(bin_precisions)';
    %%
    
    %generateCategorizedContour(data_pca,groundtruth,(1:length(point_precisions))',point_precisions,contour_file,sprintf('Areas of high test precision\nGlobal GP, confidence cutoff=%f',confidence_cutoff),[.5 .75 .9]);
    %save(sprintf('%scontourdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*100),'data_pca','groundtruth','point_precisions','contour_file','confidence_cutoff');
    
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

    for k = 1:length(fprCutoffs)
        fprintf('Overall results, FPR=%f\n',fprCutoffs(k));
        if(filter_with_confidence)
            disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
        end
        
        [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx{k},rocy{k},size(auc{k},1));
    
        [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
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

        fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points(:,k)),std(prediction_points(:,k)));

        %Display AUC Info
        disp('Individual AUCs:');
        auc_ = auc{k};
        auc_ = auc_(~isnan(auc_)); % Clear any degenerate AUC points
        disp(auc_');
        fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc_));
        disp('AUC ave err std')
        disp(mean(auc_))
        disp(std(auc_)/(length(auc_)^.5))
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
end

% disp('    TPR       FPR       TNR       FNR       cutoff');
% disp(summaryStats);

warning(orig_warn_state);

thesisCleanup;
end
