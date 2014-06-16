for energycutoffBase = 30:10:60
    for binThreshold = [10:5:20 30 40];
    %binThreshold = 30;
        disp(energycutoffBase);
        disp(binThreshold);
        GP_GraspIt_TP_hybrid_kdtree(energycutoffBase,binThreshold);
   end
end