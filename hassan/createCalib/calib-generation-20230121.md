# Environment
###    Using pfs_pipe2d w.2023.04 with following modifications:
###    pfs_utils commit d56f9d2 (tickets/INSTRM-1846)
###    And obs_pfs 6a4fac8 (master, incl PIPE2D-1138)

# Set up useful variables
    ROOTDIR=/work/drp
    CALIBDIR=/work/hassans/createCalib/CALIB-dev
    RERUNDIR=$ROOTDIR/rerun/hassans/calib/
    LOGDIR=/work/hassans/createCalib/logs

# Set up the CALIB repo and ingest all data
    mkdir -p $CALIBDIR

# Ingest defects
    makePfsDefects --lam
    ingestCuratedCalibs.py $ROOTDIR --calib $CALIBDIR $DRP_PFS_DATA_DIR/curated/pfs/defects

# Ingest sim and bootstrap detectormaps
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/detectorMap-sim-{b,r,m}{2,4}.fits --mode=copy --validity 100000 -c clobber=True
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/detectorMap-sim-n{1,3}.fits --mode=copy --validity 100000 -c clobber=True
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $DRP_PFS_DATA_DIR/detectorMap/bootstrap/*{b,r,m}{1,3}.fits --mode=copy --validity 100000 -c clobber=True

# Create biases
    constructPfsBias.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/bias --doraise --batch-type=smp --cores=1 --id visit=84579..84593 arm=b^r spectrograph=1^3

    <!-- constructPfsBias.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/bias --doraise --batch-type=smp --cores=1 --id visit=82130..82150 arm=b^r spectrograph=1^3 -->
## Ingest biases. Note that m-band calibs generated for free
    ingestPfsCalibs.py $ROOTDIR --output=$CALIBDIR --validity=1800 --doraise --mode=copy -- $RERUNDIR/bias/BIAS/*{b,r}{1,3}.fits

# Create darks
    constructPfsDark.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/dark --doraise --batch-type=smp --cores=1 --id visit=84594..84608 arm=b^r spectrograph=1^3
## Ingest darks. Note that m-band calibs generated for free
    ingestPfsCalibs.py $ROOTDIR --output=$CALIBDIR --validity=1800 --doraise --mode=copy -- $RERUNDIR/dark/DARK/*{b,r}{1,3}.fits

# Create FiberProfiles (b^r)
    nohup constructFiberProfiles.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/fiberProfiles --doraise --batch-type=none --cores=1 --id visit=82113..82127 arm=b^r spectrograph=1^3 --config isr.doFlat=False profiles.profileRadius=3 > $LOGDIR/fiberProfiles-r1b1r3b3.log 2>&1 &
## Ingest FiberProfiles, removing previous
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'delete FROM fiberProfiles WHERE arm in ("r", "b") AND spectrograph in (1, 3)'
    \rm $CALIBDIR/FIBERPROFILES/pfsFiberProfiles-*{b,r}{1,3}.fits
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $RERUNDIR/fiberProfiles/FIBERPROFILES/pfsFiberProfiles-2022-11-15-082113-*.fits  --validity 100000 --mode=copy -c clobber=True

# Create DetectorMaps (b^r)
## Using only visits in which exposures for all 4 detectors (b1, r1, b3, r3) are taken.
    nohup reduceArc.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR//detMap --id visit=81965^81966^81967^81968^81969^81970^81971^81972^81973^81974^81975^81976^81977^81978^82719^82721^82722^82723^82727^82728^83098^83100 arm=b^r spectrograph=1^3 -j 20 -c reduceExposure.isr.doFlat=False fitDetectorMap.doSlitOffsets=True > $LOGDIR/reduceArc-b1r1b3r3.log 2>&1
## Ingest DetectorMaps, removing previous
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'DELETE FROM detectorMap WHERE arm in ("r", "b") AND spectrograph in (1, 3)'
    \rm $CALIBDIR/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $RERUNDIR//detMap/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits --mode=copy --validity 100000 --config clobber=True

# FIXME add m3 visits
# Create FiberProfiles (m)
## m1
    nohup constructFiberProfiles.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/fiberProfiles_m --doraise --batch-type=none --cores=1 --id visit=80621..80640 arm=m spectrograph=1 --config isr.doFlat=False profiles.profileRadius=3 > $LOGDIR/fiberProfiles-m1.log 2>&1 &
## m3
    nohup constructFiberProfiles.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/fiberProfiles_m --doraise --batch-type=none --cores=1 --id visit=83689..83694 arm=m spectrograph=3 --config isr.doFlat=False profiles.profileRadius=3 > $LOGDIR/fiberProfiles-m3.log 2>&1 &
## Ingest FiberProfiles, removing previous
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'delete FROM fiberProfiles WHERE arm="m" AND spectrograph in (1,3)'
    \rm $CALIBDIR/FIBERPROFILES/pfsFiberProfiles-*m{1,3}.fits
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $RERUNDIR/fiberProfiles_m/FIBERPROFILES/pfsFiberProfiles-*m{1,3}.fits  --validity 100000 --mode=copy -c clobber=True

# Create DetectorMaps
## m1
    nohup reduceArc.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/detectorMap_m --id visit=79994..79997^80631..80640 arm=m spectrograph=1 -j 20 -c reduceExposure.isr.doFlat=False fitDetectorMap.doSlitOffsets=True > $LOGDIR/reduceArc-m.log 2>&1 &
## m3
    nohup reduceArc.py $ROOTDIR --calib=$CALIBDIR --rerun=$RERUNDIR/detectorMap_m --id visit=83637..83651^84090..84092 arm=m spectrograph=3 -j 20 -c reduceExposure.isr.doFlat=False fitDetectorMap.doSlitOffsets=True > $LOGDIR/reduceArc-m3.log 2>&1 &

## Ingest DetectorMaps, removing previous
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'DELETE FROM detectorMap WHERE arm = "m" AND spectrograph in (1,3)'
    \rm $CALIBDIR/DETECTORMAP/*m{1,3}.fits
    ingestPfsCalibs.py $ROOTDIR --calib $CALIBDIR $RERUNDIR/detectorMap_m/DETECTORMAP/pfsDetectorMap-*m{1,3}.fits --mode=copy --validity 100000 -c clobber=True
