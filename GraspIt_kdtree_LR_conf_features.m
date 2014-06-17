%% GraspIt_kdtree_LR_conf_features.m
%% 5/12/14
%  Builds a kdtree, trains LR classifiers for each node, then examines the
%  relative importance of each dimension for every node. This is done using
%  leave-one-out testing on precision, and comparing it to the classifier
%  trained with all dimensions.
%%
function GraspIt_kdtree_LR_conf_features(energycutoff,binThreshold)

%me = mfilename;
me = sprintf('kd_LR_features_energy_%d_bin_%d',energycutoff,binThreshold);
%use_timestamp = false;
thesisStartup;

% Hard-coded for printing convenience
metricNames = strsplit('PointArng TriangleSize Extension Spread Limit PerpSym ParallelSym OrthoNorm Volume GraspIt_Volume GraspIt_Epsilon');

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
    datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
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
      %results2 = ...
      % Physical testing
      groundtruth = [groundtruth;datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
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


data = sphereize(data);

% % Run PCA on a copy for plotting the contour
% finalDimensionNum = 2;
% [~,data_pca,latent] = pca(data);
% 
% %Reduce components based on earlier specifications
% data_pca = data_pca(:,logical([ones(1,finalDimensionNum) zeros(1,size(data,2)-finalDimensionNum)]));

% Testing only on physical testing successes
physicalCutoff = .8;

% Linear regression prediction must be >= this value to be positive
lrCutoff = .5;

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 1;

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .33;

% For diagnostic purposes - get the number of positive and negative
% training examples
trainy_total = [];

filter_with_confidence = 1;

disp(strcat('binThreshold:',num2str(binThreshold)));
disp(strcat('energy base:',num2str(energycutoff)));
disp(strcat('using confidence:',num2str(filter_with_confidence)));

auc = [];
rocx = [];
rocy = [];

for l = 1:loopn
    trainData = data;
    trainy = (results1 <= energycutoff);

    % Make testx vector of appropriate size
    if(testProportion >= 1)
        testx = zeros(testProportion,1);
    else
        testx = zeros(floor(size(trainData,1) * testProportion),1);
    end

    testIndices = randomIndices(trainData,length(testx));
    % Unlike most of the other experiments, we want predictions for
    % both test and training points.
    testx = trainData(testIndices,:);
    trainData(testIndices,:) = [];
    trainy(testIndices) = [];

    trainy_total = [trainy_total; trainy];

    dbstop if error

    % Build kdtree
    tree = kdMedCreateTree(trainData,binThreshold);
    % Don't split on results, but include them for easy bin confidence
    % computing
    bins = kdGetAllBins(tree);
    fprintf('Number of bins: %d\n',size(bins,1));
    bin_size_sum = 0;
    for i = 1:size(bins,1)
        bin_size_sum = bin_size_sum + size(bins{i,1},1);
    end
    fprintf('Average bin size: %d\n',bin_size_sum / size(bins,1));
    fprintf('\n\n');

    % Train classifiers
    nodeIndices = kdGetAllNodeIndices(tree);

    % Store which bin each training point belongs in
    trainNodes = zeros(size(trainData,1),1);
    for i = 1:size(trainData,1)
        trainNodes(i) = kdGetBinIndex(tree,trainData(i,:));
    end
    testNodes = zeros(size(testx,1),1);
    for i = 1:size(testx,1)
        testNodes(i) = kdGetBinIndex(tree,testx(i,:));
    end

    dim_precisions = cell(size(nodeIndices,1),1);
    base_precisions = cell(size(nodeIndices,1),1);
    %coefficients = cell(size(nodeIndices,1),1);
    
    % For every leaf in the tree:
    for i = 1:size(nodeIndices,1)
        prec = [(1:size(trainData,2))' zeros(size(trainData,2),1)];
        
        %% Train standard classifier and get base precision
        leafTrainingIndices = (trainNodes == nodeIndices(i));
        if sum(leafTrainingIndices) == 0
            continue;
        end
        B = glmfit(trainData(leafTrainingIndices,:), [trainy(leafTrainingIndices) ones(sum(leafTrainingIndices),1)], 'binomial', 'link', 'logit');
        
        % Predict on test set
        leafTestIndices = (testNodes == nodeIndices(i));
        testSet = testx(leafTestIndices,:);
        mf = Logistic(B(1) + testSet * B(2:end));
        res = groundtruth([false(size(trainData,1),1); leafTestIndices]);
        [base_precision, ~] = precisionAndRecall({mf > lrCutoff},{res > physicalCutoff},true);
        
        %% For every dimension:
        for d = 1:size(trainData,2)
            % Train classifier missing the specified dimension
            skimmedData = trainData;
            skimmedData(:,d) = [];
            B = glmfit(skimmedData(leafTrainingIndices,:), [trainy(leafTrainingIndices) ones(sum(leafTrainingIndices),1)], 'binomial', 'link', 'logit');
            
            % Store resulting precision in dim_precisions vector
            testSet = testx(leafTestIndices,:);
            testSet(:,d) = [];
            mf = Logistic(B(1) + testSet * B(2:end));
            %res = groundtruth([false(size(trainData,1),1); leafTestIndices]);
            [skimmed_prec, ~] = precisionAndRecall({mf > lrCutoff},{res > physicalCutoff},true);
            prec(d,2) = base_precision - skimmed_prec;
        end
        
        prec = flipud(sortrows(prec,2));
        dim_precisions{i} = prec;
        
        fprintf('Bin index %d extents:\n',nodeIndices(i));
        disp(kdGetBinExtents(tree,nodeIndices(i)));
        fprintf('Base precision: %f\n',base_precision);
        fprintf('Test points: %d\n',sum(testNodes == nodeIndices(i)));
        fprintf('Indices=');
        filteredTestIndices = testIndices(testNodes == nodeIndices(i));
        for k = 1:sum(testNodes == nodeIndices(i))
            fprintf('%d,',filteredTestIndices(k));
        end
        fprintf('\n');
        %fprintf('Precision affected by metrics:\n');
        %disp(prec);
        % TODO: add metricNames as row label
        printmat(prec,'Precision affected by metrics',strjoin(metricNames(prec(:,1))),'DimN Precision change');
        
        fprintf('End bin %d\n',nodeIndices(i));
        
    end

%     %% Generate sample contours for confidence and precision
%     if mod(l,10) == 0
%         contour_file = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_egy_%d_conf_%d_bin_%d_confidence_sample_%d',energycutoff,confidence_cutoff*10,binThreshold,l/10));
%         generateCategorizedContour(data_pca,groundtruth,(1:size(data_pca,1))',mf_confidence,contour_file,sprintf('Bins with confidence >= %f',confidence_cutoff),confidence_cutoff,true);
% 
%         contour_file = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_egy_%d_conf_%d_bin_%d_precision_sample_%d',energycutoff,confidence_cutoff*10,binThreshold,l/10));
%         generateCategorizedContour(data_pca(testIndices,:),groundtruth(testIndices),(1:size(testIndices,1))',mf_precision,contour_file,sprintf('Bins with test precision >= %f',confidence_cutoff),.5:.1:.9,true);
% 
%         clearvars test_point_confidencesl contour_file;
%     end
%     %% End generate contour

%     mfout = [mfout; mf];
%     mfout_precision = [mfout_precision; mf_precision >= confidence_cutoff]; % Precision is mapped with multiple levels for sampling so do the binary cutoff here
%     mfout_confidence = [mfout_confidence; mf_confidence >= confidence_cutoff];
%     mfout_TP = [mfout_TP; mf_TP];
%     indexout = [indexout; testIndices_final];
%     indexout_precision = [indexout_precision; testIndices];
%     indexout_confidence = [indexout_confidence; (1:size(data,1))']; % Confidence is given for all data points
%     indexout_TP = [indexout_TP; testIndices];

    dbstop if error;

%     %% Generate ROC curve
%     try
%         [perfx,perfy,perft,auc(end+1)] = perfcurve(res',normalizeer(mf)',1);
%         %rocResults = [rocResults; [perfx perfy]];
%         [rocx(:,end+1),rocy(:,end+1)] = rotandproject(perfx,perfy,.01);
%     catch err
%         auc(l) = NaN;
%     end
    %%

    disp(strcat('Completed iteration: ',num2str(l)));
%     [precision, recall] = precisionAndRecall({mf >= lrCutoff},{res},true);
%     disp(strcat('Prediction points:',num2str(size(testIndices_final,1))));
%     disp('  TP     FP     TN     FN');
%     [tp, fp, tn, fn] = getTestResults(mf >= lrCutoff,res);
%     disp([tp, fp, tn, fn]);
%     disp('Precision / Recall');
%     disp([precision recall]);
%     predictions{l} = (mf >= lrCutoff);
%     trueValues{l} = res;
end

% [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,size(auc,1));
% 
% if(filter_with_confidence)
%     contour_file = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_egy_%d_conf_%d_bin_%d',energycutoff,confidence_cutoff*10,binThreshold));
%     contour_file_TP = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_TP_egy_%d_conf_%d_bin_%d',energycutoff,confidence_cutoff*10,binThreshold));
%     contour_file_precision = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_precision_egy_%d_conf_%d_bin_%d',energycutoff,confidence_cutoff*10,binThreshold));
% else
%     contour_file = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_egy_%d_nofilter',energycutoff));
%     contour_file_TP = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_TP_egy_%d_nofilter',energycutoff));
%     contour_file_precision = strcat(savedir,'hybrid_contours\',sprintf('cntr_kd_LR_precision_egy_%d_nofilter',energycutoff));
% end
% generateCategorizedContour(data_pca,groundtruth,indexout_confidence,mfout_confidence,contour_file,sprintf('Points with high average confidence\nconfidence=%f',confidence_cutoff),[.5 .75 .9]);
% generateCategorizedContour(data_pca,groundtruth,indexout_TP,mfout_TP,contour_file_TP,sprintf('Areas of frequent true positives\nconfidence=%f',confidence_cutoff),[.5 .75 .9]);
% generateCategorizedContour(data_pca,groundtruth,indexout_precision,mfout_precision,contour_file_precision,sprintf('Areas of high average precision\nconfidence=%f',confidence_cutoff),[.5 .75 .9]);
% 
% disp('Overall results (hybrid)');
% if(filter_with_confidence)
%     disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
% end
% if(~filter_with_confidence)
%     disp('All datapoints used');
% else
%     disp('Filtered datapoints');
% end
% 
% %Display AUC Info
% disp('AUC ave err std')
% disp(mean(auc))
% disp(std(auc)/(size(auc,1))^.5)
% disp(std(auc))
% 
% [precision_macro, recall_macro] = precisionAndRecall(predictions,trueValues,true);
% disp('Precision / Recall macro');
% disp([precision_macro recall_macro]);
% disp('');
% [precision_micro, recall_micro] = precisionAndRecall(predictions,trueValues,false);
% disp('Precision / Recall micro');
% disp([precision_micro recall_micro]);

% disp('    TPR       FPR       TNR       FNR       cutoff');
% disp(summaryStats);

warning(orig_warn_state);

thesisCleanup;
end
