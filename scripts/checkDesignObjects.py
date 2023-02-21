import sys
import os
import numpy as np
from pfs.datamodel.pfsConfig import PfsDesign
from collections import Counter
import argparse

def check_design(designId, location, spectraDir):
    design = PfsDesign.read(designId, dirName=location)
    missing = False
    for catId, tract, patch, objId in zip(design.catId, design.tract, design.patch, design.objId):
        if catId == 0:
            continue
        filename = ("pfsSimObject-%05d-%05d-%s-%016x.fits" %
                    (catId, tract, patch, objId))
        path = os.path.join(spectraDir, str(catId), filename)
        exists = os.path.exists(path)
        if not exists:
            missing = True
        print(f'{designId} {catId}, {objId:#016x}, {path} {exists}')
    if missing:
        print('ERROR: Some of the objects do not have corresponding spectra.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("designIds", nargs="+", type=int)
    parser.add_argument("dirName", type=str, default='.')
    parser.add_argument("specDirName", type=str)

    args = parser.parse_args()

    for id in args.designIds:
        check_design(id, args.dirName, args.specDirName)

if __name__ == "__main__":
    main()