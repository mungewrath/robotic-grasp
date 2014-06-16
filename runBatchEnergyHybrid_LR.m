for energycutoffBase = 20:10:40
    for binThreshold = [10 30 40];
    %binThreshold = 30;
        disp(energycutoffBase);
        disp(binThreshold);
        GraspIt_hybrid_kdtree_LR_conf(energycutoffBase,binThreshold);
   end
end