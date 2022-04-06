Sky Line Lists
==============

Subject to [PIPE2D-1017](https://pfspipe.ipmu.jp/jira/browse/PIPE2D-1017) being merged, these data are in DRP-readable (`ReferenceLineSet.fromLineList()`) format. See the headers for each linelist for column details.

1. [rousselot-linelist.txt](rousselot-linelist.txt): derived from the Rousselot+2000 [linelist](https://people.ast.cam.ac.uk/~optics/dazle/sky-data/Rousellot_list_v2.0.dat) format. The `source` column is set to `20` in all cases.
2. [osterbrock-linelist.txt](osterbrock-linelist.txt): derived from Osterbrock+1996 and Osterbrock+1997. Original data available from [CDS](https://cdsarc.cds.unistra.fr/viz-bin/cat/III/211#/browse). The `source=30` in all cases.
3. [rousselot-osterbrock-merged-linelist.txt](rousselot-osterbrock-merged-linelist.txt): combined linelist from 1. and 2. above. Lines from the individual catalogs are merged if they are within 0.05 nm of each other. When merging, the Rousselot data are used, with the exception of the transitions which are taken from the Osterbrock data. The `source` is set to `40` for merged lines, otherwise the original source information from the input catalogs are used.
