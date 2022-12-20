from opdb import opdb
import pandas as pd

def main():
    visits = getVisits()

    #  visits = [1,2,3,5,7,8,10]
    ranges = process(visits)

    print(f'There are {len(ranges)} groups found: {ranges}')
    convert(ranges)

def getVisits():
    db = opdb.OpDB(hostname='pfsa-db01',
                port=5432,
                dbname='opdb',
                username='pfs')

    result = db.bulkSelect('iic_sequence',
                        "select visit_set.pfs_visit_id, "
                        "iic_sequence.iic_sequence_id, "
                        "iic_sequence.sequence_type, "
                        " iic_sequence.name from iic_sequence "
                        "inner join visit_set "
                        "on iic_sequence.iic_sequence_id = visit_set.iic_sequence_id "
                        "where iic_sequence.sequence_type='domeFlat' "
                        "order by visit_set.pfs_visit_id asc")

    return result['pfs_visit_id']


def process(visits):

    ranges = []
    visitStart = None
    visitEnd = None
    for count, visit in visits.iteritems():
    # for visit in visits:
        if not visitStart:
            visitStart = visit
            visitEnd = visit
            continue
        if visit - visitEnd == 1:
            visitEnd = visit
        else:
            handle(visitStart, visitEnd, ranges)
            visitStart = visit
            visitEnd = visit
    handle(visitStart, visitEnd, ranges)
    return ranges


def handle(visitStart, visitEnd, ranges):
    if visitEnd == visitStart:
        ranges.append((visitStart, None))
        return
    if visitEnd - visitStart == 1:
        ranges.append((visitStart, None))
        ranges.append((visitEnd, None))
        return
    ranges.append((visitStart, visitEnd))


def convert(ranges):
    fileName = 'snippet.py'
    with open(fileName, 'w') as file:
        for visitStart, visitEnd in ranges:
            if not visitEnd:
                file.write(f'self.add("S", [{visitStart}], W_AITQTH=True)\n')
            else:
                file.write(f'self.add("S", list(range({visitStart}, {visitEnd} + 1)), W_AITQTH=True)\n')
    print(f'Code snippet written out to {fileName}.')

if __name__ == "__main__":
    main()
