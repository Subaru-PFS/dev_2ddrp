# Set up useful variables
WORKDIR=/scratch/hassans/sim4convert-PIPE2D-1088b

# Get PFSA (CCD) sim NIR data for core weekly and science
mkdir $WORKDIR/raw

## Check number of PFFA NIR files. The spectrograph=1 and armNum=3 for NIR (n1)
    $ ls /projects/HSC/PFS/weekly-20210819/PFFA*13.fits|wc -l
    59
## Copy them over
    cp /projects/HSC/PFS/weekly-20210819/PFFA*13.fits $WORKDIR/raw/.
## Copy over pfsDesign/Config files also
    cp /projects/HSC/PFS/weekly-20210819/pfs*.fits $WORKDIR/raw/.
## Check numbers
    $ ls -l $WORKDIR/raw/PFFA*.fits|wc -l
    59
    $ ls -l $WORKDIR/raw/pfs*.fits|wc -l
    63

## Science data
mkdir $WORKDIR/raw-science
cp -P /projects/HSC/PFS/scienceSims/scienceSims-20210908/*13.fits $WORKDIR/raw-science/.
$ ls -l $WORKDIR/raw-science
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
cp /projects/HSC/PFS/scienceSims/scienceSims-20210908/pfs*.fits $WORKDIR/raw-science/.


# Detrend PFFA data.
## This is to get the geometry correct (flip over amp images) and to correct for the bias and dark, as up-the-ramp data not have biases, and darks are assumed to be negligible.

## Create basic repo
repo=$WORKDIR/repo1
mkdir $repo
mkdir $repo/CALIB
echo "lsst.obs.pfs.PfsMapper" > $repo/_mapper

## Need to use old DRP that allows processing of PFSA NIR data. This is w.2022.40a
setup pfs_pipe2d w.2022.40a

## Ingest images into repo
ingestPfsImages.py $repo --mode=link $WORKDIR/raw/PFFA*.fits -c clobber=True register.ignore=True
ingestPfsImages.py $repo --mode=link $WORKDIR/raw-science/PFFA*.fits -c clobber=True register.ignore=True

ingestCuratedCalibs.py "$repo" --calib "$repo"/CALIB "$DRP_PFS_DATA_DIR"/curated/pfs/defects
### Note: version of drp_pfs_data_dir used: w.2022.48, but it shoudn't matter what was used.

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

python $WORK_DIR/software/dev_2ddrp/scripts/convertPFSAtoPFSB.py --inDir  /projects/HSC/PFS/weekly-20210819 --ccdDir $repo/rerun/detrend/postIsrCcd/2020-01-01 --visits 0..58 --outDir $WORKDIR/raw > /scratch/hassans/log/convertPFSAtoPFSB-20221130a.log 2>&1 &

python $WORK_DIR/software/dev_2ddrp/scripts/convertPFSAtoPFSB.py --inDir /projects/HSC/PFS/scienceSims/scienceSims-20210908 --ccdDir $repo/rerun/detrend/postIsrCcd/2020-01-01 --visits 1000..1007 --outDir $WORKDIR/raw-science > /scratch/hassans/log/convertPFSAtoPFSB-20221130b.log 2>&1 &

# setup to latest weekly
setup pfs_pipe2d w.2022.48

# And development version of pfs_pipe2d and obs_pfs
/tigress/hassans/software/obs_pfs[tickets/PIPE2D-1088%<>] $ setup -jr .
scons
/tigress/hassans/software/pfs_pipe2d[u/hassans/20221130a%] $ setup -jr .
scons

# Run weekly core
nohup $PFS_PIPE2D_DIR/weekly/process_weekly.sh -d $WORKDIR/raw -c 1 /scratch/hassans/weekly/weekly-pipe2d-1088c > /scratch/hassans/log/weekly-pipe2d-1088-20221201a.log 2>&1 &
# Run weekly science
