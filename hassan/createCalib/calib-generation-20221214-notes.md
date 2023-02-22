# Start with CALIB containing bootstrap data for b1, r1, b3, r3, m1 and m3
# and no fiberProfiles for the above detectors.

    CALIBDIR=/work/hassans/createCalib/calib-b1r1b3r3/CALIB

# Ingest bootstrap detectormaps, removing existing detectormaps, if present
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'DELETE FROM detectorMap WHERE arm in ("r", "b", "m") AND spectrograph in (1, 3)'
    \rm $CALIBDIR/DETECTORMAP/pfsDetectorMap-*{b,r,m}{1,3}.fits
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR /work/hassans/software/drp_pfs_data/detectorMap/bootstrap/*{b,r,m}{1,3}.fits --mode=copy --validity 100000 -c clobber=True

# b1, r1, b3, r3: these can use same fiberTrace and scienceArc exposures

## Generate fiberProfiles
    nohup constructFiberProfiles.py /work/drp --calib=/work/hassans/createCalib/calib-b1r1b3r3/CALIB --rerun=hassans/r1b1r3b3-fiberProfiles --doraise --batch-type=none --cores=1 --id visit=82113..82127 arm=b^r spectrograph=1^3 --config isr.doFlat=False profiles.profileRadius=3 >/work/hassans/fiberProfiles-r1b1r3b3.log 2>&1 &
## Ingest fiberProfiles
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'delete FROM fiberProfiles WHERE arm in ("r", "b", "m") AND spectrograph in (1, 3)'
    \rm $CALIBDIR/FIBERPROFILES/pfsFiberProfiles-*{b,r,m}{1,3}.fits
    ingestPfsCalibs.py /work/drp --calib /work/hassans/createCalib/calib-b1r1b3r3/CALIB /work/drp/rerun/hassans/r1b1r3b3-fiberProfiles/FIBERPROFILES/pfsFiberProfiles-2022-11-15-082113-*.fits  --validity 100000 --mode=copy -c clobber=True

## Detectormap
Using only visits in which exposures for all 4 detectors (b1, r1, b3, r3) are taken.
    nohup reduceArc.py /work/drp --calib=/work/hassans/createCalib/calib-b1r1b3r3/CALIB --rerun=hassans/b1r1b3r3-detectorMap4 --id visit=81965^81966^81967^81968^81969^81970^81971^81972^81973^81974^81975^81976^81977^81978^82719^82721^82722^82723^82727^82728^83098^83100 arm=b^r spectrograph=1^3 -j 20 -c reduceExposure.isr.doFlat=False fitDetectorMap.doSlitOffsets=True > /work/hassans/logs/reduceArc-arc-quartz-b1r1b3r3-20221215a.log 2>&1
### Stats
    reduceArc.fitDetectorMap INFO: Final fit: chi2=56332904.501217 dof=16139946 xRMS=0.060827 yRMS=0.058533 (0.003991 nm) xSoften=0.059518 ySoften=0.000000 from 16123413/16174210 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=12106174.718731 xRMS=0.060469 yRMS=0.062400 (0.004254 nm) xSoften=0.084095 ySoften=0.000000 from 1797134 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=16325660.212783 dof=16139946 xRMS=0.061083 yRMS=0.054814 (0.003737 nm) xSoften=0.059510 ySoften=0.000000 from 1797134 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=3529349.178859 xRMS=0.060458 yRMS=0.062667 (0.004273 nm) xSoften=0.084084 ySoften=0.000000 from 1797134 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=16488022.011436 dof=16138920 xRMS=0.032908 yRMS=0.031614 xSoften=0.030375 ySoften=0.000000 from 16123413/16123413 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 31 fiberIds: [1, 3, 23, 45, 84, 85, 86, 92, 114, 137, 184, 229, 273, 280, 309, 315, 316, 336, 359, 367, 382, 417, 426, 471, 487, 515, 560, 601, 603, 607, 651]
    reduceArc.fitDetectorMap INFO: Final fit: chi2=16471484.565706 dof=16139910 xRMS=0.032892 yRMS=0.030578 (0.002085 nm) xSoften=0.030357 ySoften=0.000000 from 16123376/16174210 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=7667746.058007 xRMS=0.033531 yRMS=0.031330 (0.002136 nm) xSoften=0.066838 ySoften=0.000000 from 1797134 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=16172467.610550 dof=16139910 xRMS=0.032906 yRMS=0.022673 (0.001546 nm) xSoften=0.030357 ySoften=0.000000 from 1797134 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=7528642.599592 xRMS=0.033533 yRMS=0.031839 (0.002171 nm) xSoften=0.066837 ySoften=0.000000 from 1797134 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=16471873.558022 dof=16138884 xRMS=0.032892 yRMS=0.031519 xSoften=0.030357 ySoften=0.000000 from 16123376/16123376 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 31 fiberIds: [1, 3, 23, 45, 84, 85, 86, 92, 114, 137, 184, 229, 273, 280, 309, 315, 316, 336, 359, 367, 382, 417, 426, 471, 487, 515, 560, 601, 603, 607, 651]
    reduceArc.fitDetectorMap INFO: Final result: chi2=16471833.872353 dof=16138740 xRMS=0.032892 yRMS=0.031519 xSoften=0.030357 ySoften=0.000000 from 16123376 lines
    reduceArc.fitDetectorMap INFO: Stats for CdI: chi2=11742.658648 dof=9406 xRMS=0.056868 yRMS=0.037146 xSoften=0.026567 ySoften=0.000000 from 4703 lines
    reduceArc.fitDetectorMap INFO: Stats for HgI: chi2=0.000000 dof=0 xRMS=nan yRMS=nan xSoften=0.000000 ySoften=0.000000 from 0 lines
    reduceArc.fitDetectorMap INFO: Stats for KrI: chi2=2111.764907 dof=2084 xRMS=0.070579 yRMS=0.073325 xSoften=0.000000 ySoften=0.000000 from 1042 lines
    reduceArc.fitDetectorMap INFO: Stats for NeI: chi2=24091.932550 dof=21866 xRMS=0.047453 yRMS=0.027605 xSoften=0.026199 ySoften=0.004788 from 10933 lines
    reduceArc.fitDetectorMap INFO: Stats for Trace: chi2=16433887.516249 dof=16106698 xRMS=0.032876 yRMS=nan xSoften=0.030364 ySoften=0.000000 from 16106698 lines
    reduceArc.fitDetectorMap INFO: Stats for fiberId=2: chi2=46030.742438 dof=27475 xRMS=0.043287 yRMS=0.038056 xSoften=0.041785 ySoften=0.000000 from 27447 lines (2 KrI, 18 NeI, 27419 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=161: chi2=28755.426316 dof=27602 xRMS=0.033449 yRMS=0.033800 xSoften=0.030750 ySoften=0.000000 from 27575 lines (2 KrI, 17 NeI, 27548 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=338: chi2=32927.430827 dof=27312 xRMS=0.035352 yRMS=0.038595 xSoften=0.033453 ySoften=0.000000 from 27284 lines (2 KrI, 18 NeI, 27256 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=495: chi2=27281.092534 dof=26440 xRMS=0.034085 yRMS=0.035932 xSoften=0.030585 ySoften=0.000000 from 26411 lines (2 KrI, 19 NeI, 26382 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=650: chi2=36600.460192 dof=27996 xRMS=0.036923 yRMS=0.033207 xSoften=0.034929 ySoften=0.000000 from 27970 lines (1 KrI, 17 NeI, 27944 Trace, 8 CdI)
    reduceArc INFO: Writing output for {'visit': 81965, 'arm': 'b', 'spectrograph': 1, 'dateObs': '2022-11-15', 'site': 'S', 'category': 'A', 'field': 'M39', 'ccd': 0, 'filter': 'b', 'expTime': 30.0, 'dataType': 'COMPARISON', 'taiObs': '2022-11-15T05:51:18.5
    63', 'pfsDesignId': 9212402307179353158, 'slitOffset': 0.0}
    reduceArc.fitDetectorMap INFO: Final fit: chi2=84800385.481022 dof=16460363 xRMS=0.073200 yRMS=0.069366 (0.004730 nm) xSoften=0.071605 ySoften=0.000000 from 16444125/16502668 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=14812179.407924 xRMS=0.063013 yRMS=0.078034 (0.005321 nm) xSoften=0.091419 ySoften=0.000000 from 1833630 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=16679583.492272 dof=16460363 xRMS=0.072730 yRMS=0.063771 (0.004348 nm) xSoften=0.071602 ySoften=0.000000 from 1833630 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=3148477.264607 xRMS=0.062981 yRMS=0.078577 (0.005358 nm) xSoften=0.091419 ySoften=0.000000 from 1833630 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=25273567.850434 dof=16459303 xRMS=0.039964 yRMS=0.034306 xSoften=0.038172 ySoften=0.000000 from 16444119/16444125 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1303, 1347, 1394, 1439, 1486, 1531, 1575, 1588, 1618, 1638, 1669, 1684, 1728, 1773, 1817, 1862, 1909, 1924, 1945, 1953]
    reduceArc.fitDetectorMap INFO: Final fit: chi2=25262422.349347 dof=16460428 xRMS=0.039955 yRMS=0.033478 (0.002283 nm) xSoften=0.038164 ySoften=0.000000 from 16444190/16502668 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=8076287.223135 xRMS=0.042280 yRMS=0.035852 (0.002445 nm) xSoften=0.067871 ySoften=0.000000 from 1833630 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=16488721.108528 dof=16460428 xRMS=0.040051 yRMS=0.022938 (0.001564 nm) xSoften=0.038163 ySoften=0.000000 from 1833630 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=5464487.723186 xRMS=0.042290 yRMS=0.033949 (0.002315 nm) xSoften=0.067870 ySoften=0.000000 from 1833630 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=25263120.319095 dof=16459370 xRMS=0.039956 yRMS=0.034308 xSoften=0.038164 ySoften=0.000000 from 16444185/16444190 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1303, 1347, 1394, 1439, 1486, 1531, 1575, 1588, 1618, 1638, 1669, 1684, 1728, 1773, 1817, 1862, 1909, 1924, 1945, 1953]
    reduceArc.fitDetectorMap INFO: Final result: chi2=25263316.960480 dof=16459236 xRMS=0.039956 yRMS=0.034540 xSoften=0.038164 ySoften=0.000000 from 16444190 lines
    reduceArc.fitDetectorMap INFO: Stats for CdI: chi2=6710.031306 dof=9344 xRMS=0.038918 yRMS=0.039772 xSoften=0.000000 ySoften=0.000000 from 4672 lines
    reduceArc.fitDetectorMap INFO: Stats for HgI: chi2=0.000000 dof=0 xRMS=nan yRMS=nan xSoften=0.000000 ySoften=0.000000 from 0 lines
    reduceArc.fitDetectorMap INFO: Stats for KrI: chi2=1823.501483 dof=2106 xRMS=0.060878 yRMS=0.066833 xSoften=0.000000 ySoften=0.000000 from 1053 lines
    reduceArc.fitDetectorMap INFO: Stats for NeI: chi2=20432.062974 dof=21314 xRMS=0.041306 yRMS=0.031437 xSoften=0.019806 ySoften=0.002380 from 10657 lines
    reduceArc.fitDetectorMap INFO: Stats for Trace: chi2=25234351.364716 dof=16427808 xRMS=0.039955 yRMS=nan xSoften=0.038188 ySoften=0.000000 from 16427808 lines
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1304: chi2=50454.946524 dof=28114 xRMS=0.042850 yRMS=0.027091 xSoften=0.041473 ySoften=0.000000 from 28085 lines (1 KrI, 19 NeI, 28056 Trace, 9 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1459: chi2=49057.802695 dof=27430 xRMS=0.043678 yRMS=0.039306 xSoften=0.041788 ySoften=0.000000 from 27404 lines (2 KrI, 16 NeI, 27378 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1616: chi2=44463.916296 dof=27584 xRMS=0.041034 yRMS=0.047696 xSoften=0.039254 ySoften=0.000000 from 27571 lines (2 KrI, 2 NeI, 27558 Trace, 9 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1794: chi2=39517.728574 dof=27715 xRMS=0.038531 yRMS=0.040765 xSoften=0.036648 ySoften=0.000000 from 27690 lines (2 KrI, 15 NeI, 27665 Trace, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1952: chi2=43009.642674 dof=27649 xRMS=0.040302 yRMS=0.041078 xSoften=0.038502 ySoften=0.000000 from 27623 lines (2 KrI, 17 NeI, 27597 Trace, 7 CdI)
    reduceArc INFO: Writing output for {'visit': 81965, 'arm': 'b', 'spectrograph': 3, 'dateObs': '2022-11-15', 'site': 'S', 'category': 'A', 'field': 'M39', 'ccd': 6, 'filter': 'b', 'expTime': 30.0, 'dataType': 'COMPARISON', 'taiObs': '2022-11-15T05:51:12.3
    27', 'pfsDesignId': 9212402307179353158, 'slitOffset': 0.0}
    reduceArc.fitDetectorMap INFO: Final fit: chi2=63454777.303944 dof=17182698 xRMS=0.059739 yRMS=0.055586 (0.004800 nm) xSoften=0.059163 ySoften=0.000000 from 17138191/17237698 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=17235546.607873 xRMS=0.059357 yRMS=0.065743 (0.005677 nm) xSoften=0.087979 ySoften=0.029562 from 1915300 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=18018399.375907 dof=17182698 xRMS=0.059886 yRMS=0.050527 (0.004363 nm) xSoften=0.059163 ySoften=0.000000 from 1915300 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=27368685.990670 xRMS=0.059354 yRMS=0.066156 (0.005713 nm) xSoften=0.087978 ySoften=0.029535 from 1915300 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=22298026.286767 dof=17181670 xRMS=0.035417 yRMS=0.030347 xSoften=0.034452 ySoften=0.000000 from 17138190/17138191 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 31 fiberIds: [1, 3, 23, 45, 84, 85, 86, 92, 114, 137, 184, 229, 273, 280, 309, 315, 316, 336, 359, 367, 382, 417, 426, 471, 487, 515, 560, 601, 603, 607, 651]
    reduceArc.fitDetectorMap INFO: Final fit: chi2=22304572.643310 dof=17182896 xRMS=0.035422 yRMS=0.029807 (0.002574 nm) xSoften=0.034458 ySoften=0.000000 from 17138394/17237875 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=12678208.681682 xRMS=0.038266 yRMS=0.034619 (0.002990 nm) xSoften=0.073814 ySoften=0.029505 from 1915319 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=17297136.461354 dof=17182896 xRMS=0.035472 yRMS=0.018686 (0.001614 nm) xSoften=0.034458 ySoften=0.000000 from 1915319 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=31504047.737414 xRMS=0.038272 yRMS=0.035040 (0.003026 nm) xSoften=0.073814 ySoften=0.029495 from 1915319 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=22305325.539454 dof=17181868 xRMS=0.035422 yRMS=0.030289 xSoften=0.034458 ySoften=0.000000 from 17138393/17138394 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 31 fiberIds: [1, 3, 23, 45, 84, 85, 86, 92, 114, 137, 184, 229, 273, 280, 309, 315, 316, 336, 359, 367, 382, 417, 426, 471, 487, 515, 560, 601, 603, 607, 651]
    reduceArc.fitDetectorMap INFO: Final result: chi2=22305311.027146 dof=17181721 xRMS=0.035422 yRMS=0.030307 xSoften=0.034458 ySoften=0.000000 from 17138389 lines
    reduceArc.fitDetectorMap INFO: Stats for CdI: chi2=7440.532936 dof=9462 xRMS=0.036756 yRMS=0.030405 xSoften=0.014466 ySoften=0.000000 from 4731 lines
    reduceArc.fitDetectorMap INFO: Stats for HgI: chi2=1474.045847 dof=950 xRMS=0.112504 yRMS=0.160310 xSoften=0.000000 ySoften=0.033383 from 475 lines
    reduceArc.fitDetectorMap INFO: Stats for HgII: chi2=0.000000 dof=0 xRMS=nan yRMS=nan xSoften=0.000000 ySoften=0.000000 from 0 lines
    reduceArc.fitDetectorMap INFO: Stats for KrI: chi2=9944.285840 dof=16060 xRMS=0.028968 yRMS=0.025498 xSoften=0.013834 ySoften=0.009257 from 8030 lines
    reduceArc.fitDetectorMap INFO: Stats for NeI: chi2=45215.604485 dof=62820 xRMS=0.038297 yRMS=0.031040 xSoften=0.014152 ySoften=0.002948 from 31410 lines
    reduceArc.fitDetectorMap INFO: Stats for Trace: chi2=22241236.558037 dof=17093743 xRMS=0.035421 yRMS=nan xSoften=0.034518 ySoften=0.000000 from 17093743 lines
    reduceArc.fitDetectorMap INFO: Stats for fiberId=2: chi2=74074.674342 dof=29133 xRMS=0.049749 yRMS=0.046282 xSoften=0.049034 ySoften=0.000000 from 29066 lines (13 KrI, 28999 Trace, 45 NeI, 8 CdI, 1 HgI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=161: chi2=38297.496982 dof=29374 xRMS=0.035427 yRMS=0.030781 xSoften=0.034519 ySoften=0.000000 from 29300 lines (13 KrI, 29226 Trace, 53 NeI, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=338: chi2=56787.550536 dof=28858 xRMS=0.043626 yRMS=0.037551 xSoften=0.042871 ySoften=0.000000 from 28791 lines (12 KrI, 28724 Trace, 46 NeI, 9 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=495: chi2=35818.203558 dof=29159 xRMS=0.034858 yRMS=0.027016 xSoften=0.033527 ySoften=0.000000 from 29086 lines (9 KrI, 29013 Trace, 56 NeI, 8 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=650: chi2=61026.343049 dof=29437 xRMS=0.044734 yRMS=0.033993 xSoften=0.044040 ySoften=0.000000 from 29365 lines (15 KrI, 29293 Trace, 47 NeI, 9 CdI, 1 HgI)
    reduceArc INFO: Writing output for {'visit': 81965, 'arm': 'r', 'spectrograph': 1, 'dateObs': '2022-11-15', 'site': 'S', 'category': 'A', 'field': 'M39', 'ccd': 1, 'filter': 'r', 'expTime': 30.0, 'dataType': 'COMPARISON', 'taiObs': '2022-11-15T05:51:18.5
    72', 'pfsDesignId': 9212402307179353158, 'slitOffset': 0.0}
    reduceArc.fitDetectorMap INFO: Final fit: chi2=88917547.717576 dof=17178309 xRMS=0.070385 yRMS=0.070342 (0.006077 nm) xSoften=0.069868 ySoften=0.000000 from 17128047/17280106 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=19453826.381592 xRMS=0.060447 yRMS=0.074182 (0.006409 nm) xSoften=0.096555 ySoften=0.021646 from 1920012 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=19234667.923382 dof=17178309 xRMS=0.070487 yRMS=0.068604 (0.005927 nm) xSoften=0.069864 ySoften=0.000000 from 1920012 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=14742920.298016 xRMS=0.060456 yRMS=0.074874 (0.006469 nm) xSoften=0.096532 ySoften=0.021715 from 1920012 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=25988256.649057 dof=17175006 xRMS=0.038061 yRMS=0.032946 xSoften=0.037340 ySoften=0.000000 from 17125793/17128047 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1303, 1347, 1394, 1439, 1486, 1531, 1575, 1588, 1618, 1638, 1669, 1684, 1728, 1773, 1817, 1862, 1909, 1924, 1945, 1953]
    reduceArc.fitDetectorMap INFO: Final fit: chi2=25973563.700251 dof=17177561 xRMS=0.038051 yRMS=0.032285 (0.002789 nm) xSoften=0.037333 ySoften=0.000000 from 17127931/17280516 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=12496940.954645 xRMS=0.038881 yRMS=0.036683 (0.003169 nm) xSoften=0.076632 ySoften=0.021304 from 1920057 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=17331276.323300 dof=17177561 xRMS=0.038187 yRMS=0.018972 (0.001639 nm) xSoften=0.037332 ySoften=0.000000 from 1920057 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=18366216.128178 xRMS=0.038903 yRMS=0.036919 (0.003190 nm) xSoften=0.076621 ySoften=0.021367 from 1920057 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=25972698.069093 dof=17175886 xRMS=0.038050 yRMS=0.032752 xSoften=0.037331 ySoften=0.000000 from 17127305/17127931 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1303, 1347, 1394, 1439, 1486, 1531, 1575, 1588, 1618, 1638, 1669, 1684, 1728, 1773, 1817, 1862, 1909, 1924, 1945, 1953]
    reduceArc.fitDetectorMap INFO: Final result: chi2=25971752.183123 dof=17175735 xRMS=0.038049 yRMS=0.032764 xSoften=0.037331 ySoften=0.000000 from 17127297 lines
    reduceArc.fitDetectorMap INFO: Stats for CdI: chi2=43883.647172 dof=7432 xRMS=0.119356 yRMS=0.028301 xSoften=0.082589 ySoften=0.009587 from 3716 lines
    reduceArc.fitDetectorMap INFO: Stats for HgI: chi2=5180.514201 dof=2638 xRMS=0.133744 yRMS=0.151326 xSoften=0.000000 ySoften=0.040759 from 1319 lines
    reduceArc.fitDetectorMap INFO: Stats for HgII: chi2=0.000000 dof=0 xRMS=nan yRMS=nan xSoften=0.000000 ySoften=0.000000 from 0 lines
    reduceArc.fitDetectorMap INFO: Stats for KrI: chi2=39259.768920 dof=19408 xRMS=0.063815 yRMS=0.024453 xSoften=0.041742 ySoften=0.009556 from 9704 lines
    reduceArc.fitDetectorMap INFO: Stats for NeI: chi2=195924.571742 dof=70070 xRMS=0.096098 yRMS=0.034403 xSoften=0.053367 ySoften=0.004249 from 35035 lines
    reduceArc.fitDetectorMap INFO: Stats for Trace: chi2=25687503.681089 dof=17077523 xRMS=0.037896 yRMS=nan xSoften=0.037239 ySoften=0.000000 from 17077523 lines
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1304: chi2=61659.860454 dof=28778 xRMS=0.045052 yRMS=0.043304 xSoften=0.044575 ySoften=0.000000 from 28695 lines (13 KrI, 28617 Trace, 60 NeI, 7 CdI, 3 HgI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1459: chi2=54163.422988 dof=28670 xRMS=0.042496 yRMS=0.029443 xSoften=0.041837 ySoften=0.000000 from 28587 lines (16 KrI, 28511 Trace, 60 NeI, 7 CdI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1616: chi2=60837.190595 dof=28253 xRMS=0.045364 yRMS=0.041284 xSoften=0.044677 ySoften=0.000000 from 28179 lines (14 KrI, 28105 Trace, 53 NeI, 5 CdI, 2 HgI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1794: chi2=38688.692260 dof=28766 xRMS=0.035815 yRMS=0.028323 xSoften=0.035071 ySoften=0.000000 from 28683 lines (18 KrI, 28600 Trace, 55 NeI, 7 CdI, 3 HgI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=1952: chi2=77305.124660 dof=27264 xRMS=0.052502 yRMS=0.042847 xSoften=0.052282 ySoften=0.000000 from 27176 lines (20 KrI, 27088 Trace, 57 NeI, 9 CdI, 2 HgI)
    reduceArc INFO: Writing output for {'visit': 81965, 'arm': 'r', 'spectrograph': 3, 'dateObs': '2022-11-15', 'site': 'S', 'category': 'A', 'field': 'M39', 'ccd': 7, 'filter': 'r', 'expTime': 30.0, 'dataType': 'COMPARISON', 'taiObs': '2022-11-15T05:51:12.4
    63', 'pfsDesignId': 9212402307179353158, 'slitOffset': 0.0}
## Ingest new detectormaps, removing previous
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'DELETE FROM detectorMap WHERE arm in ("r", "b") AND spectrograph in (1, 3)'
    \rm $CALIBDIR/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR /work/drp/rerun/hassans/b1r1b3r3-detectorMap4/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits --mode=copy --validity 100000 -c clobber=True

# m1 detector

## FiberProfiles
    nohup constructFiberProfiles.py /work/drp --calib=/work/hassans/createCalib/calib-b1r1b3r3/CALIB --rerun=hassans/r1b1r3b3-fiberProfiles --doraise --batch-type=none --cores=1 --id visit=80621..80640 arm=m spectrograph=1 --config isr.doFlat=False profiles.profileRadius=3 >/work/hassans/fiberProfiles-m1.log 2>&1 &

### Ingest
    ingestPfsCalibs.py /work/drp --calib /work/hassans/createCalib/calib-b1r1b3r3/CALIB /work/drp/rerun/hassans/r1b1r3b3-fiberProfiles/FIBERPROFILES/pfsFiberProfiles-2022-09-27-080631-m1.fits  --validity 100000 --mode=copy -c clobber=True

## Detectormap
    nohup reduceArc.py /work/drp --calib=/work/hassans/createCalib/calib-b1r1b3r3/CALIB --rerun=hassans/m1-detectorMap --id visit=79994..79997^80631..80640 arm=m spectrograph=1 -j 20 -c reduceExposure.isr.doFlat=False fitDetectorMap.doSlitOffsets=True > /work/hassans/logs/reduceArc-arc-quartz-m1-20221214a.log 2>&1 &

### Stats
    reduceArc.fitDetectorMap INFO: Final fit: chi2=55234286.878507 dof=21468264 xRMS=0.053570 yRMS=0.053230 (0.002511 nm) xSoften=0.052389 ySoften=0.000000 from 21457654/22468657 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=25746329.200016 xRMS=0.056757 yRMS=0.051321 (0.002421 nm) xSoften=0.141549 ySoften=0.000000 from 2496518 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=21572977.624547 dof=21468264 xRMS=0.054752 yRMS=0.039921 (0.001883 nm) xSoften=0.052340 ySoften=0.000000 from 2496518 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=12237938.703781 xRMS=0.057022 yRMS=0.049063 (0.002315 nm) xSoften=0.141023 ySoften=0.000000 from 2496518 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=20563689.167696 dof=21428966 xRMS=0.032705 yRMS=0.027782 xSoften=0.029126 ySoften=0.000000 from 21419404/21457654 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1, 3, 45, 92, 137, 184, 229, 273, 280, 309, 316, 336, 359, 382, 426, 471, 515, 560, 607, 651]
    reduceArc.fitDetectorMap INFO: Final fit: chi2=21266225.128738 dof=21481500 xRMS=0.033234 yRMS=0.026852 (0.001267 nm) xSoften=0.029780 ySoften=0.000000 from 21470860/22481452 lines
    reduceArc.fitDetectorMap INFO: Fit quality from reserved lines: chi2=22124833.718399 xRMS=0.028590 yRMS=0.026992 (0.001273 nm) xSoften=0.135971 ySoften=0.000000 from 2497939 lines (10.0%)
    reduceArc.fitDetectorMap INFO: Softened fit: chi2=21507357.545371 dof=21481500 xRMS=0.033199 yRMS=0.017097 (0.000807 nm) xSoften=0.029780 ySoften=0.000000 from 2497939 lines
    reduceArc.fitDetectorMap INFO: Softened fit quality from reserved lines: chi2=22397232.096520 xRMS=0.028580 yRMS=0.026895 (0.001269 nm) xSoften=0.135982 ySoften=0.000000 from 2497939 lines
    reduceArc.fitDetectorMap INFO: Slit offsets measurement: chi2=20517973.076942 dof=21440499 xRMS=0.032662 yRMS=0.027480 xSoften=0.029066 ySoften=0.000000 from 21430907/21470860 lines
    reduceArc.fitDetectorMap INFO: Unable to measure slit offsets for 20 fiberIds: [1, 3, 45, 92, 137, 184, 229, 273, 280, 309, 316, 336, 359, 382, 426, 471, 515, 560, 607, 651]
    reduceArc.fitDetectorMap INFO: Final result: chi2=21271758.307753 dof=21479853 xRMS=0.033238 yRMS=0.027480 xSoften=0.029786 ySoften=0.000000 from 21470405 lines
    reduceArc.fitDetectorMap INFO: Stats for ArI: chi2=5753.555240 dof=5726 xRMS=0.043519 yRMS=0.021854 xSoften=0.025696 ySoften=0.007194 from 2863 lines
    reduceArc.fitDetectorMap INFO: Stats for KrI: chi2=2217.741789 dof=2500 xRMS=0.038266 yRMS=0.020095 xSoften=0.024254 ySoften=0.007878 from 1250 lines
    reduceArc.fitDetectorMap INFO: Stats for NeI: chi2=11458.676972 dof=10632 xRMS=0.056256 yRMS=0.032130 xSoften=0.024397 ySoften=0.000000 from 5316 lines
    reduceArc.fitDetectorMap INFO: Stats for OH: chi2=4.590560 dof=8 xRMS=0.030562 yRMS=0.122703 xSoften=0.000000 ySoften=0.000000 from 4 lines
    reduceArc.fitDetectorMap INFO: Stats for Trace: chi2=21249750.770260 dof=21459621 xRMS=0.033231 yRMS=nan xSoften=0.029790 ySoften=0.000000 from 21459621 lines
    reduceArc.fitDetectorMap INFO: Stats for XeI: chi2=2572.972931 dof=2702 xRMS=0.049801 yRMS=0.035605 xSoften=0.019151 ySoften=0.006272 from 1351 lines
    reduceArc.fitDetectorMap INFO: Stats for fiberId=2: chi2=69993.277619 dof=37398 xRMS=0.045008 yRMS=0.029647 xSoften=0.043371 ySoften=0.000000 from 37381 lines (9 NeI, 37364 Trace, 3 KrI, 1 XeI, 4 ArI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=158: chi2=61584.262181 dof=36926 xRMS=0.047169 yRMS=0.028445 xSoften=0.048104 ySoften=0.000000 from 36908 lines (8 NeI, 36890 Trace, 3 KrI, 2 XeI, 5 ArI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=337: chi2=84514.312149 dof=36109 xRMS=0.052195 yRMS=0.037734 xSoften=0.049615 ySoften=0.000000 from 36090 lines (9 NeI, 36071 Trace, 3 KrI, 2 XeI, 5 ArI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=494: chi2=49788.725086 dof=37487 xRMS=0.038495 yRMS=0.020689 xSoften=0.036539 ySoften=0.000000 from 37470 lines (10 NeI, 37453 Trace, 1 KrI, 2 XeI, 4 ArI)
    reduceArc.fitDetectorMap INFO: Stats for fiberId=650: chi2=114020.184269 dof=22579 xRMS=0.074987 yRMS=0.026961 xSoften=0.078156 ySoften=0.000000 from 22564 lines (7 NeI, 22549 Trace, 1 KrI, 2 XeI, 5 ArI)


### Ingest, removing previous bootstrap
    sqlite3 $CALIBDIR/calibRegistry.sqlite3 'DELETE FROM fiberprofiles WHERE arm = "m" AND spectrograph = 1'
    \rm $CALIBDIR/DETECTORMAP/*m1.fits
    ingestPfsCalibs.py /work/drp --calib $CALIBDIR /work/drp/rerun/hassans/m1-detectorMap/DETECTORMAP/pfsDetectorMap-061162-m1.fits --mode=copy --validity 100000 -c clobber=True

### Test
    reduceExposure.py /work/drp --calib /work/drp/CALIB --output /work/rhl/rerun/rhl/2022-11 -c doAdjustDetectorMap=False doMeasureLines=False extractSpectra.doCrosstalk=False isr.doFlat=False --id visit=83259  arm=b^r spectrograph=1^3

### Copy calibs to official CALIB directory
    sqlite3 /work/drp/CALIB/calibRegistry.sqlite3 'delete FROM fiberProfiles WHERE arm in ("r", "b") AND spectrograph in (1, 3)'
    rm -i /work/drp/CALIB/FIBERPROFILES/pfsFiberProfiles-*{b,r}{1,3}.fits
    ingestPfsCalibs.py /work/drp --calib /work/drp/CALIB /work/drp/rerun/hassans/r1b1r3b3-fiberProfiles/FIBERPROFILES/pfsFiberProfiles-2022-11-15-082113-*.fits  --validity 100000 --mode=copy -c clobber=True

    sqlite3 /work/drp/CALIB/calibRegistry.sqlite3 'DELETE FROM detectorMap WHERE arm in ("r", "b") AND spectrograph in (1, 3)'
    rm -i /work/drp/CALIB/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits
    ingestPfsCalibs.py /work/drp --calib /work/drp/CALIB /work/drp/rerun/hassans/b1r1b3r3-detectorMap4/DETECTORMAP/pfsDetectorMap-*{b,r}{1,3}.fits --mode=copy --validity 100000 -c clobber=True
