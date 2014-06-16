for energycutoffBase = 20:10:40
    for binThreshold = [10 30 40];
    %binThreshold = 30;
        disp(energycutoffBase);
        disp(binThreshold);
        GraspIt_LR_global(energycutoffBase,binThreshold);
   end
end