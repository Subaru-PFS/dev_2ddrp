import re

def handle(match):
    return f'self.inclRange({match.group(1)}, {int(match.group(2))-1})'

def main():
    """Code for replacing range entries in headerFixes.py
    with calls to new function inclRange().
    """
    regex1 = r"list\(range\((\d+)\,\s*(\d+)\s*\+\s+1\)\s*\)"
    subst1 = "self.inclRange(\\1, \\2)"
    regex2 = r"list\(range\((\d+)\,\s*(\d+)\s*\)\s*\)"
    regex3 = r"range\((\d+)\,\s*(\d+)\s*\)"
    with open('/work/hassans/software/pfs_utils/python/pfs/utils/headerFixes.py', 'r') as file:
        with open('out.py', 'w') as out:
            for line in file.readlines():
                result1, nsubs = re.subn(regex1, subst1, line, 0, re.MULTILINE)
                if nsubs >= 1:
                    out.write(result1)
                else:
                    result2, nsubs = re.subn(regex2, handle, line, 0, re.MULTILINE)
                    if nsubs >= 1:
                        out.write(result2)
                    else:
                        result3, nsubs = re.subn(regex3, handle, line, 0, re.MULTILINE)
                        out.write(result3)


if __name__ == "__main__":
    main()
