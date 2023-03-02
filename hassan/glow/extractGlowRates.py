import sys
import argparse
import re
import ast
import pandas as pd

PATTERN = r".*\((\{.*\})\).*Scaled.*\s([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)\/second\)"

def process(inFileName, outFileName):


    with open(inFileName, 'r') as inFile:

        data = []
        for line in inFile:
            match = re.match(PATTERN, line)
            if match:
                metadataGroup = match.group(1)
                metadata = ast.literal_eval(metadataGroup)
                visit = metadata['visit']
                taiObs = metadata['taiObs']
                dataType = metadata['dataType']
                scale = float(match.group(2))
                rate = float(match.group(3))
                data.append((visit, taiObs, dataType, scale, rate))

        df = pd.DataFrame(data, columns=['visit', 'taiObs', 'dataType', 'scale', 'rate'])
        df.to_csv(outFileName, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str, help="log file to extract rates")

    args = parser.parse_args()

    process(args.logfile, "out.csv")

if __name__ == "__main__":
    main()

# ".*\{\'visit'\:\s*([0-9]{5}).*\'dateObs\'\:\s*\'([0-9]{4})\-([0-9]{2})-([0-9]{2})\'.*\'arm\'\:\s*\'([a-z])'.*\'spectrograph\'\:\s*([0-9]).*Scaled.*\s([0-9]+\.[0-9]+)\s*\(([0-9]+\.[0-9]+)\/second\)"
