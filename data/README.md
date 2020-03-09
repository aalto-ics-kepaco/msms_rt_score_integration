# General Information

### ```use_inchi```

While constructing the database, the structure representation used to calculate
molecular fingerprints or descriptors are InChIs. Even though the database 
contains SMILES (canonical and isomeric), those SMILES are calcualted from the
InChI representation using [RDKit Python package](http://rdkit.org/docs/api-docs.html):

1. [MolFromInChI](http://rdkit.org/docs/source/rdkit.Chem.inchi.html#rdkit.Chem.inchi.MolFromInchi) 
2. MolToSmiles

This protocal was chosen as some retention time databases only contain InChI representations.