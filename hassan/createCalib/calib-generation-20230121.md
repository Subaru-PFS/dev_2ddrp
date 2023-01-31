# Set up useful variables
    ROOTDIR=/work/drp
    CALIBDIR=/work/hassans/createCalib/CALIB-dev

# Set up the CALIB repo and ingest all data
    mkdir -p $CALIBDIR

# Ingest defects
    makePfsDefects --lam
    ingestCuratedCalibs.py $ROOTDIR --calib $CALIBDIR $DRP_PFS_DATA_DIR/curated/pfs/defects

# Ingest sim and bootstrap detectormaps
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/detectorMap-sim-{b,r,m}{2,4}.fits --mode=copy --validity 100000 -c clobber=True
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/detectorMap-sim-n{1,3}.fits --mode=copy --validity 100000 -c clobber=True
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/bootstrap/*{b,r,m}{1,3}.fits --mode=copy --validity 100000 -c clobber=True

# Create biases
    <!-- constructPfsBias.py $ROOTDIR --calib=$CALIBDIR --rerun=$ROOTDIR/rerun/hassans/calibs/bias --doraise --batch-type=smp --cores=1 --id visit=84579..84593 arm=b^r spectrograph=1^3 -->

    constructPfsBias.py $ROOTDIR --calib=$CALIBDIR --rerun=$ROOTDIR/rerun/hassans/calibs/bias --doraise --batch-type=smp --cores=1 --id visit=82130..82150 arm=b^r spectrograph=1^3

## Ingest biases. Note that m-band calibs generated for free
    ingestPfsCalibs.py $ROOTDIR --output=$CALIBDIR --validity=1800 --doraise --mode=copy -- $ROOTDIR/rerun/hassans/calibs/bias/BIAS/*{b,r}{1,3}.fits

# Create darks
    constructPfsDark.py $ROOTDIR --calib=$CALIBDIR --rerun=$ROOTDIR/rerun/hassans/calibs/dark --doraise --batch-type=smp --cores=1 --id visit=82151..82171 arm=b^r spectrograph=1^3

## Ingest darks. Note that m-band calibs generated for free
    ingestPfsCalibs.py $ROOTDIR --output=$CALIBDIR --validity=1800 --doraise --mode=copy -- $ROOTDIR/rerun/hassans/calibs/dark/DARK/*{b,r}{1,3}.fits

# Create FiberProfiles


# Create DetectorMaps
