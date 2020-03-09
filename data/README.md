# (MS2, RT)-tuple Database

All data used in our experiments is organized in SQLite DB. The database can be downloaded from 
[Zenodo](ADD_LINK_HERE) (~12GB), and its layout and as well as content is described here.

## General Information

The database contains

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
