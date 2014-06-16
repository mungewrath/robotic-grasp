import sys

#dimensions = [2, 3]
confidences = [str(i*.1) for i in range(5,10)]
binSizes = [30, 40, 50, 60]
energyLevels = [30, 40, 50, 60]

AUCs = {}
errs = {}
precisions = {}
recalls = {}

# Open file, write headers
categories = ''
for e in energyLevels:
    categories += ','*3 + 'energy_{0},'.format(e)
categories += ','
for e in energyLevels:
    categories += ','*3 + 'energy_{0} (baseline),'.format(e)
header = 'BinSize,Confidence,' + 'AUC,err,precision,recall,'*len(energyLevels) + ',' + 'AUC,err,precision,recall,'*len(energyLevels)

with open(sys.argv[1],'w') as outfile:
    outfile.write(categories+'\n')
    outfile.write(header+'\n')
    for e in energyLevels:
        # Process baseline
        f = open('LR_energy_{0}.log'.format(e),'r')
        l = f.readline()
        while 'AUC ave' not in l:
            l = f.readline()
        baseKey = (e)
        # Get AUC
        l = f.readline()
        AUCs[baseKey] = float(l)
        # Get error
        f.readline()
        l = f.readline()
        errs[baseKey] = float(l)
        # Get precision and recall
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        l = f.readline()
        pr = l.split()
        precisions[baseKey] = float(pr[0])
        recalls[baseKey] = float(pr[1])
        f.close()

        for b in binSizes:
            fname = 'kd_LR_energy_{0}_bin_{1}.log'.format(e,b)
            f = open(fname,'r')
            for c in confidences:
                l = f.readline()
                while 'Confidence threshold' not in l:
                    l = f.readline()
                key = (e,b,c)
                # Expecting a line in the format "Confidence threshold=0.5"
                print l.rstrip()[-3:]
                print c
                assert l.rstrip()[-3:] == c
                # Skip 3 lines, read AUC
                f.readline()
                f.readline()
                l = f.readline()
                AUCs[key] = float(l)
                # Skip 2 lines, read err
                f.readline()
                l = f.readline()
                errs[key] = float(l)
                # Skip 8 lines, read precision + recall
                for i in range(7):
                    f.readline()
                l = f.readline()
                pr = l.split()
                precisions[key] = float(pr[0])
                recalls[key] = float(pr[1])
            f.close()

    # Write 
    for b in binSizes:
        for c in confidences:
            outfile.write('{0},{1},'.format(b,c))
            # Handle confidence data
            for e in energyLevels:
                key = (e,b,c)
                outfile.write('{0},{1},{2},{3},'.format(AUCs[key],errs[key],precisions[key],recalls[key]))
            outfile.write(',')
            # Handle non-filtered data
            for e in energyLevels:
                key = (e)
                outfile.write('{0},{1},{2},{3},'.format(AUCs[key],errs[key],precisions[key],recalls[key]))
            outfile.write('\n')
