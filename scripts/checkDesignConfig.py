import sys
import numpy as np
from pfs.datamodel.pfsConfig import PfsDesign
from collections import Counter
import argparse

def check_design(design_id, location):
    design = PfsDesign.read(design_id, dirName=location)

    counts = Counter(zip(design.catId, design.objId))
    if counts.most_common(1)[0][1] > 1:
        duplicates = {tup: count for tup, count in counts.items() if count > 1}
        print(f'design {design.pfsDesignId:#016x} contains duplicate occurrences of'
              ' the same (catId, objId) combination. Details below:\n'
              f'{{(catId, objId): number of occurrences}}:\n\t {duplicates}')

    dup = duplicates(design)
    if len(dup) > 0:
        print(f'design {design_id:#016x} contains duplicate occurrences tuple: {{(catId, tract, patch, objId): n_occurrences}}:\n\t {dup}')
    else:
        print(f'design {design_id:#016x} contains NO duplicate occurrences of tuple (catId, tract, patch, objId)')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("designIds", nargs="+", type=int)
    parser.add_argument("dirName", type=str, default='.')

    args = parser.parse_args()

    for id in args.designIds:
        check_design(id, args.dirName)

if __name__ == "__main__":
    main()