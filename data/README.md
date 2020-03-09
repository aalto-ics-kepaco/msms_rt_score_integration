# (MS2, RT)-tuple Database

All data used in our experiments is organized in SQLite DB. The database can be downloaded from 
[Zenodo](ADD_LINK_HERE) (~12GB), and its layout and as well as content is described here.   

## General Information

The database was build around the [publicly available MS2 scores](https://sourceforge.net/p/casmi/web/HEAD/tree/web/2016/contest/submissions/) 
of the various metabolite identification approaches that participated in the 
[CASMI 2016 challenge](http://www.casmi-contest.org/2016/index.shtml) for a set of 208 MS2 spectra. 
Later, the MS2 data, used by [Ruttkies et al. 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9)
for the development of their MS2 + RT score integration framework, was added to the database as well. 

### (MS2, RT)-tuples

The table ```challenges``` contains four entries corresponding to the datasets used in our publication
(see Section 3.1 and Table 1): CASMI 2016, **positive** and **negative**; Massbank EA, **EA_positive** and **EA_negative**.

To each challenge, a set of (MS2, RT)-tuples is associated and can be found in table ```spectra```. 

TODO: Either add the spectra-files to the repository, or (better), add the (mass, intensity)-tuples directly into
the DB.



The RankSVM training retention times are _not_ part of this database, but rather the preference 
scores predicted by the RankSVM models (see Section 3.2), for the molecular candidate structures.

## Database Layout 

![database_layout](/data/db_layout.png)

### ```use_inchi```

While constructing the database, the structure representation used to calculate
molecular fingerprints or descriptors are InChIs. Even though the database 
contains SMILES (canonical and isomeric), those SMILES are calcualted from the
InChI representation using [RDKit Python package](http://rdkit.org/docs/api-docs.html):

1. [MolFromInChI](http://rdkit.org/docs/source/rdkit.Chem.inchi.html#rdkit.Chem.inchi.MolFromInchi) 
2. MolToSmiles

This protocal was chosen as some retention time databases only contain InChI representations.
