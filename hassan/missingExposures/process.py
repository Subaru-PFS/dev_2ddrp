from astropy.io import fits
import re

regexRaw = ".*PFSA([0-9]{6})([0-9])([0-9]).fits"
regexIngested = ".*PFSA([0-9]{6})([a-z][0-9]).fits"


def main():
    # site='hilo'
    site='pu'
    visitStart = 81866 # 2022-11-14 exposure
    # visitStart = 84574 # ALF suggestion

    rawVisitDict, rawVisitFileDict = readRawVisits(site, visitStart)
    print(f'rawVisits={len(rawVisitDict)}')

    ingestedVisitDict = readIngestedVisits(site)
    print(f'ingestedVisits={len(ingestedVisitDict)}')

    problemVisits = findProblemVisits(rawVisitDict, ingestedVisitDict)
    groupVisits(problemVisits)
    checkDesign(site, problemVisits, rawVisitFileDict)

def readRawVisits(site, visitStart):
    arm = {1: 'b', 2:'r', 3:'n', 4:'m'}
    rawVisitDict = dict()
    rawVisitFileDict = dict()
    with open(f'raw-exposures-{site}.txt', 'r') as f:
        for file in f:
            match = re.search(regexRaw, file)
            if match:
                visit = int(match.group(1))
                spectrograph = match.group(2)
                armNum = int(match.group(3))
                detector = f'{arm[armNum]}{spectrograph}'
                if visit < visitStart:
                    continue

                if not visit in rawVisitFileDict:
                    rawVisitFileDict[visit] = []
                rawVisitFileDict[visit].append(file[:-1])

                if not visit in rawVisitDict:
                    rawVisitDict[visit] = [detector]
                else:
                    detectors = rawVisitDict[visit]
                    detectors.append(detector)
                    rawVisitDict[visit] = detectors
    return rawVisitDict, rawVisitFileDict

def readIngestedVisits(site):
    ingestedVisitDict = dict()
    with open(f"ingested_exposures_{site}.txt", 'r') as g:
        for filename in g:
            match = re.search(regexIngested, filename)
            if match:
                visit = int(match.group(1))
                detector = match.group(2)
                if visit not in ingestedVisitDict:
                    ingestedVisitDict[visit] = [detector]
                else:
                    detectors = ingestedVisitDict[visit]
                    detectors.append(detector)
                    ingestedVisitDict[visit] = detectors
    return ingestedVisitDict


def findProblemVisits(rawVisitDict, ingestedVisitDict):
    problemVisits = set()
    for visit, rawDetectors in rawVisitDict.items():
        if not visit in ingestedVisitDict:
            print(f'{visit} is in raw visit list, but not in ingested list')
            continue
        detectors = ingestedVisitDict[visit]
        if len(detectors) < len(rawDetectors):
            print(f'Ingestion problem found with visit {visit}: raw detectors = {rawDetectors}, '
                  f'but ingested = {detectors}')
            problemVisits.add(visit)

    print(f'There are {len(problemVisits)} problematic visits: {problemVisits}')
    return problemVisits

def groupVisits(problemVisits):
    print(f'There are {len(problemVisits)} problematic visits.')
    if len(problemVisits) == 0:
        print('No problematic visits to provide fixHeader or query for.')
        return
    # Group contiguous visits from problemVisit set
    print('Grouping problem visits:')
    groups = []
    startValue = None
    endValue = None
    problemVisits = sorted(problemVisits)
    for ii in problemVisits:
        if not startValue:
            startValue = ii
            endValue = ii
            continue
        # print(f'ii:{ii}, startValue:{startValue} endValue:{endValue}')
        if ii == endValue + 1:
            endValue = ii
            continue
        groups.append((startValue, endValue))
        startValue = ii
        endValue = ii
    groups.append((startValue, endValue))
    print(f'Number of groups: {len(groups)}')
    print(groups)
    print()
    print("Add this to pfs_utils headerFixes.py:\n")
    for start, end in groups:
        print(f"self.add(\"S\", self.inclRange({start}, {end}), W_PFDSGN=0x5cab8319135e443f)")

    query = "select * from raw where "
    first = True
    for start, end in groups:
        if first:
            query += f"(visit between {start} and {end})"
            first = False
            continue
        query += f" OR (visit between {start} and {end})"

    query += ";"
    print(f"Use this to query the registry:\n{query}")

def checkDesign(site, problemVisit, rawVisitFileDict):
    with open(f'visitDesign_{site}.csv', 'w') as l:

        for visit in problemVisit:
            print(f'Checking for SuNSS exp for visit {visit}..')
            if not visit in rawVisitFileDict:
                print(f'visit {visit} not in rawExposuresDict')
                continue
            foundSuNSS = False
            foundNonSuNSS = False
            for file in rawVisitFileDict[visit]:
                hdul = fits.open(file)
                if hdul[0].header['W_PFDSGN'] == 0xdeadbeef:
                    foundSuNSS = True
                else:
                    foundNonSuNSS = True
                l.write(f"{visit} {hex(hdul[0].header['W_PFDSGN'])}, {file}\n")
            assert foundSuNSS, f'Visit {visit}: no SuNSS exposure found.'
            assert foundNonSuNSS, f'Visit {visit}: no Non-SuNSS exposure found.'

if __name__ == "__main__":
    main()
