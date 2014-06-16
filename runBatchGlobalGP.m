for energycutoffBase = 20:10:40
    for binSize = [10 30 40]
        disp(energycutoffBase);
        disp(binSize);
        GP_precision_baseline(energycutoffBase,binSize);
    end
end