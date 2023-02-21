import sys
import argparse
import os

def processCommand(parser, count, commandString, repeatCommands):
    args = parser.parse_args(commandString.split())
    for v in args.visit:
        imageName = getRawImageName(v, args.detector)
        if not os.path.exists(imageName):
            print(f'visit={v}, detector={args.detector}: {imageName} doesnt exist')
            repeatCommands.add(commandString)

def getRawImageName(visit, detector):
    # PFFA00001113.fits
    arm = {'b': 1, 'r': 2, 'n': 3, 'm':4}[detector[0]]
    spectrograph = detector[1]
    return f'PFFA{visit:06}{spectrograph}{arm}.fits'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', action='store', required=True, help="Detector name, e.g., r1")
    parser.add_argument('--visit', type=int, action="append", required=True, help="Visit number")
    parser.add_argument('-p', '--pfsDesignId', type=int, required=True, help="pfsDesignId")
    parser.add_argument('--dirName', default=".", help="Directory in which to write")
    parser.add_argument('--objSpectraDir', nargs='?',
                        help="Directory from which to read object spectra")
    parser.add_argument('--catConfig', default='catalog_config.yaml',
                        help=("Name of catalog config file for object spectra."
                              "Location of file defined by --objSpectraDir"))
    parser.add_argument('--pfsConfig', default=False, action="store_true", help="Generate pfsConfig?")
    parser.add_argument('--type', required=True, choices=("bias", "dark", "flat", "arc", "object"),
                        help="Type of image")
    parser.add_argument('--exptime', action='store', default=0.0, type=float)
    parser.add_argument('--xoffset', action='store', type=float, default=0.0,
                    help='shift in slit position along slit, in microns')
    parser.add_argument('--imagetyp', help="Value for IMAGETYP header")
    parser.add_argument('--lamps', default="", help="List of lamps that are on (QUARTZ,NE,HG,XE,CD,KR)")


    parser.add_argument('command')

    with open('command.log', 'r') as file1:
        lines = file1.readlines()
        count = 0
        repeatCommands=set()
        for line in lines:
            if line.startswith('makeSim'):
                processCommand(parser, count, line, repeatCommands)
                count += 1
        if repeatCommands:
            print('Repeat the following commands:')
            for command in repeatCommands:
                print(command)
        else:
            print('All sim files created OK.')


if __name__ == "__main__":
    main()