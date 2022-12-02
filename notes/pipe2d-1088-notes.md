# Set up useful variables
    ROOTDIR=/scratch/hassans/sim4convert-PIPE2D-1088b

# Get PFFA (CCD) sim NIR data for core weekly and science
## Core weekly data
### Create directory
    mkdir $ROOTDIR/raw

### Check number of PFFA NIR files. The spectrograph=1 and armNum=3 for NIR (n1)
    $ ls /projects/HSC/PFS/weekly-20210819/PFFA*13.fits|wc -l
    59
### Copy them over
    cp /projects/HSC/PFS/weekly-20210819/PFFA*13.fits $ROOTDIR/raw/.
### Copy over pfsDesign/Config files also
    cp /projects/HSC/PFS/weekly-20210819/pfs*.fits $ROOTDIR/raw/.
### Check numbers
    $ ls -l $ROOTDIR/raw/PFFA*.fits|wc -l
    59
    $ ls -l $ROOTDIR/raw/pfs*.fits|wc -l
    63

## Science data
    mkdir $ROOTDIR/raw-science
    cp -P /projects/HSC/PFS/scienceSims/scienceSims-20210908/*13.fits $ROOTDIR/raw-science/.
    $ ls -l $ROOTDIR/raw-science
    total 88552
    -rw-rw-r--. 1 hassans astro 25050240 Nov 30 23:01 PFFA00100013.fits
    -rw-rw-r--. 1 hassans astro 25050240 Nov 30 23:01 PFFA00100113.fits
    lrwxrwxrwx. 1 hassans astro       17 Nov 30 23:01 PFFA00100213.fits -> PFFA00100013.fits
    lrwxrwxrwx. 1 hassans astro       17 Nov 30 23:01 PFFA00100313.fits -> PFFA00100113.fits
    -rw-rw-r--. 1 hassans astro 20286720 Nov 30 23:01 PFFA00100413.fits
    -rw-rw-r--. 1 hassans astro 20286720 Nov 30 23:01 PFFA00100513.fits
    lrwxrwxrwx. 1 hassans astro       17 Nov 30 23:01 PFFA00100613.fits -> PFFA00100413.fits
    lrwxrwxrwx. 1 hassans astro       17 Nov 30 23:01 PFFA00100713.fits -> PFFA00100513.fits

### Config files
    cp /projects/HSC/PFS/scienceSims/scienceSims-20210908/pfs*.fits $ROOTDIR/raw-science/.


# Detrend PFFA data.
This is to get the geometry correct (flip over amp images) and to correct for the bias and dark, as up-the-ramp data not have biases, and darks are assumed to be negligible.

## Create basic repo
    repo=$ROOTDIR/repo1
    mkdir $repo
    mkdir $repo/CALIB
    echo "lsst.obs.pfs.PfsMapper" > $repo/_mapper

## Need to use old DRP that allows processing of PFSA NIR data. This is w.2022.40a
    setup pfs_pipe2d w.2022.40a
## Make sure drp_pfs_data is local and writable
    cd $WORK_DIR/software/drp_pfs_data
    setup -j -r .

## Ingest images into repo
    ingestPfsImages.py $repo --mode=link $ROOTDIR/raw/PFFA*.fits -c clobber=True register.ignore=True
    ingestPfsImages.py $repo --mode=link $ROOTDIR/raw-science/PFFA*.fits -c clobber=True register.ignore=True

    ingestCuratedCalibs.py "$repo" --calib "$repo"/CALIB "$DRP_PFS_DATA_DIR"/curated/pfs/defects
(Note: version of drp_pfs_data_dir used: w.2022.48, but it shoudn't matter what was used.)

### Generate biases, darks aand flats
    constructPfsBias.py "$repo" --calib "$repo"/CALIB --rerun calib/bias --id field=BIAS arm=n --cores 10
    ingestPfsCalibs.py "$repo" --output=$repo/CALIB --validity=1800 --doraise --mode=copy -- $repo/rerun/calib/bias/BIAS/*.fits

    constructPfsDark.py "$repo" --calib "$repo"/CALIB --rerun calib/dark --id field=DARK arm=n --cores 10
    ingestPfsCalibs.py "$repo" --output=$repo/CALIB --validity=1800 --doraise --mode=copy -- $repo/rerun/calib/dark/DARK/*.fits

    constructFiberFlat.py "$repo" --calib "$repo"/CALIB --rerun calib/flat --id field=FLAT arm=n --cores 10
    ingestPfsCalibs.py "$repo" --output=$repo/CALIB --validity=1800 --doraise --mode=copy -- $repo/rerun/calib/flat/FLAT/*.fits

## Now run detrend on all visits
    nohup detrend.py "$repo" --calib $repo/CALIB --id visit=0..58^1000..1007 arm=n --rerun detrend -c doRepair=False -j10 > /scratch/hassans/simh4convert/detrend-20221130a.log 2>&1 &

## Convert detrended files to PFSB file format

    python $WORK_DIR/software/dev_2ddrp/scripts/convertPFSAtoPFSB.py --inDir  /projects/HSC/PFS/weekly-20210819 --ccdDir $repo/rerun/detrend/postIsrCcd/2020-01-01 --visits 0..58 --outDir $ROOTDIR/raw > /scratch/hassans/log/convertPFSAtoPFSB-20221130a.log 2>&1 &

    python $WORK_DIR/software/dev_2ddrp/scripts/convertPFSAtoPFSB.py --inDir /projects/HSC/PFS/scienceSims/scienceSims-20210908 --ccdDir $repo/rerun/detrend/postIsrCcd/2020-01-01 --visits 1000..1007 --outDir $ROOTDIR/raw-science > /scratch/hassans/log/convertPFSAtoPFSB-20221130b.log 2>&1 &

# setup to latest weekly
    setup pfs_pipe2d w.2022.48
## Make sure drp_pfs_data is local and writable
    cd $WORK_DIR/software/drp_pfs_data
    setup -j -r .
## And development version of pfs_pipe2d and obs_pfs
    /tigress/hassans/software/obs_pfs[tickets/PIPE2D-1088%<>] $ setup -jr .
    scons
    /tigress/hassans/software/pfs_pipe2d[u/hassans/20221130a%] $ setup -jr .
    scons

## Consruct new directories for raw and science data, replacing PFxA n-band data with PFxB format data
    cp -r /projects/HSC/PFS/weekly-20210819 $ROOTDIR/raw-out
    \rm $ROOTDIR/raw-out/PFFA*13.fits
    cp $ROOTDIR/raw/PFFB*.fits $ROOTDIR/raw-out

    cp -r /projects/HSC/PFS/scienceSims/scienceSims-20210908 $ROOTDIR/raw-science-out
    \rm $ROOTDIR/raw-science-out/PFFA*13.fits
    cp $ROOTDIR/raw-science/PFFB*.fits $ROOTDIR/raw-science-out

## Check contents of new directories
### Raw
    ls /projects/HSC/PFS/weekly-20210819 | wc -l
    240
    ls $ROOTDIR/raw-out | wc -l
    240
    $ ls /projects/HSC/PFS/weekly-20210819/PFFA*13.fits|wc -l
    59
    $ ls $ROOTDIR/raw-out/PFFB*13.fits | wc -l
    59

### Science
    $ ls /projects/HSC/PFS/scienceSims/scienceSims-20210908|wc -l
    35
    $ ls $ROOTDIR/raw-science-out|wc -l
    35
    $ ls /projects/HSC/PFS/scienceSims/scienceSims-20210908/PFFA*13.fits|wc -l
    8
    $ ls $ROOTDIR/raw-science-out/PFFB*13.fits|wc -l
    8

# Run weekly core and weekly science
## Weekly core
    nohup $PFS_PIPE2D_DIR/weekly/process_weekly.sh -c 10 /scratch/hassans/weekly/weekly-pipe2d-1088f > /scratch/hassans/log/weekly-pipe2d-1088-20221201f.log 2>&1 &# Run weekly science

## Science
    nohup $PFS_PIPE2D_DIR/weekly/process_science.sh -c 10 /scratch/hassans/weekly/weekly-pipe2d-1088f > /scratch/hassans/log/weekly-science-pipe2d-1088-20221201f.log 2>&1 &

# Move new raw data to /projects
    cp -r /scratch/hassans/sim4convert-PIPE2D-1088b/raw-out /projects/HSC/PFS/weekly-20221201
    cp -r /scratch/hassans/sim4convert-PIPE2D-1088b/raw-science-out /projects/HSC/PFS/scienceSims/scienceSims-20221201
