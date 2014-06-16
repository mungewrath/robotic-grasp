for energycutoffBase = 30:10:60
    for binThreshold = 30:10:60;
    %binThreshold = 30;
        disp(energycutoffBase);
        disp(binThreshold);
        GraspIt_hybrid_kdtree_LR_conf_lasso(energycutoffBase,binThreshold);
   end
end