import sys
import re
import os
from pathlib import Path

# Datamodel specifies the following format:
#    pfsSimObject-%05d-%05d-%s-%016x.fits" % (catId, tract, patch, objId)
expectedPattern = re.compile(r"pfsSimObject-(\d{5})-(\d{5})-(\d+\,\d+)-([\da-f]{16}).fits")


def checkFileName(fileName):

    match = re.match(expectedPattern, fileName)
    return bool(match)


def main(args):

    if not args:
        raise ValueError("Need to specify directory to read from.")

    rootDir = args[0]

    validFiles = 0
    inValidFiles = 0
    for oldPath in Path(rootDir).glob('**/pfsSimObject*.fits'):
        _, file = os.path.split(oldPath)
        isValid = checkFileName(file)
        print(f'file {file} matches expected format: {isValid}')
        if isValid:
            validFiles += 1
        else:
            inValidFiles += 1
    print(f'Number of valid files: {validFiles}')
    print(f'Number of inValid files: {inValidFiles}')


if __name__ == "__main__":
    main(sys.argv[1:])
