import re

regex = ".*PFSA([0-9]{6})([a-z][0-9]).fits"


def main():
    ingestedVisitDict = dict()
    problemVisits = set()
    with open("raw-exposures-hilo.txt", 'r') as f, open("visits_since_nov22.csv", "r") as g, open("missing_exposures-hilo.csv", "w") as h:
        for filename in f:
            match = re.search(regex, filename)
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
            pfsDesignId = int(p)
            if visit not in ingestedVisitDict:
                print(f'visit {visit} not in ingested list.')
                continue
            detectors = ingestedVisitDict[visit]
            detector = f'{arm}{spectrograph}'
            if detector not in detectors:
                h.write(f'{visit}, {arm}, {spectrograph}, {detector}, {hex(pfsDesignId)}, {timeExpStart}')
                problemVisits.add(visit)
        # print('Problem visits:')
        # for v in sorted(problemVisits):
        #     print(v)
        # Group contiguous visits from problemVisit set
        print('Grouping problem visits:')
        groups = []
        startValue = None
        endValue = None
        problemVisits = sorted(problemVisits)
        for ii in problemVisits:
            if ii < 84574:
                continue
            if not startValue:
                startValue = ii
                endValue = ii
                continue
            # print(f'ii:{ii}, startValue:{startValue} endValue:{endValue}')
            if ii == endValue + 1:
                endValue = ii
                continue
            groups.append((startValue, endValue))
            startValue = None
            endValue = None
        groups.append((startValue, endValue))
        print(f'Number of groups: {len(groups)}')
        print(groups)
        print()
        print("Add this to pfs_utils headerFixes.py:\n")
        for start, end in groups:
            print(f"self.add(\"S\", self.inclRange({start}, {end}), W_PFDSGN=0x5cab8319135e443f)")


if __name__ == "__main__":
    main()
