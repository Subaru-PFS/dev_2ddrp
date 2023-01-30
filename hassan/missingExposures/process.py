from astropy.io import fits
import re

regex1 = ".*PFSA([0-9]{6})([a-z][0-9]).fits"
regex2 = ".*PFSA([0-9]{6})([0-9]{2}).fits"


def main():
    ingestedVisitDict = dict()
    problemVisits = set()
    with open("ingested_exposures_hilo.txt", 'r') as f, open("visits_since_nov22.csv", "r") as g, open("missing_exposures_hilo.csv", "w") as h:
        for filename in f:
            match = re.search(regex1, filename)
            if match:
                visit = int(match.group(1))
                detector = match.group(2)
                if visit not in ingestedVisitDict:
                    ingestedVisitDict[visit] = [detector]
                else:
                    detectors = ingestedVisitDict[visit]
                    detectors.append(detector)
                    ingestedVisitDict[visit] = detectors

        first = True
        for line in g:
            if first:
                first = False
                continue
            v, arm, spectrograph, p, timeExpStart = line.split(',')
            visit = int(v)
            if visit < 84574:
                continue
            pfsDesignId = int(p)
            if visit not in ingestedVisitDict:
                # print(f'visit {visit} not in ingested list.')
                continue
            detectors = ingestedVisitDict[visit]
            detector = f'{arm}{spectrograph}'
            if detector not in detectors:
                h.write(f'{visit}, {arm}, {spectrograph}, {detector}, {hex(pfsDesignId)}, {timeExpStart}')
                problemVisits.add(visit)
        print(f'There are {len(problemVisits)} problematic visits.')
        print(problemVisits)
        with open('visits-expected.txt', 'w') as h:
            for v in sorted(problemVisits):
                h.write(f'{v}\n')
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

        rawExposuresDict = dict()
        with open('raw-exposures-hilo.txt', 'r') as j:
            for file in j:
                match = re.search(regex2, file)
                if match:
                    visit = int(match.group(1))
                    if not visit in rawExposuresDict:
                        rawExposuresDict[visit] = []
                    rawExposuresDict[visit].append(file[:-1])

        with open('problematic-visits-hilo.csv', 'r') as k, open('visitDesign.csv', 'w') as l:
            for v in k:
                visit = int(v)
                print(f'Checking for SuNSS exp for visit {visit}..')
                if not visit in rawExposuresDict:
                    print(f'visit {visit} not in rawExposuresDict')
                    continue
                foundSuNSS = False
                foundNonSuNSS = False
                for file in rawExposuresDict[visit]:
                    hdul = fits.open(file)
                    if hdul[0].header['W_PFDSGN'] == 0xdeadbeef:
                        foundSuNSS = True
                    else:
                        foundNonSuNSS = True
                    l.write(f"{visit} {hex(hdul[0].header['W_PFDSGN'])}, {file}\n")
                assert(foundSuNSS, f'Visit {visit}: no SuNSS exposure found.')
                assert(foundNonSuNSS, f'Visit {visit}: no Non-SuNSS exposure found.')


if __name__ == "__main__":
    main()
