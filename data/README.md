# (MS2, RT)-tuple Database

All data used in our experiments is organized in SQLite DB. The database can be
downloaded from Zenodo](ADD_LINK_HERE) (~12GB). Its layout as well as content is
described here.   

## General Information

The database was build around the [publicly available MS2 scores](https://sourceforge.net/p/casmi/web/HEAD/tree/web/2016/contest/submissions/)
of the various metabolite identification approaches that participated in the
[CASMI 2016 challenge](http://www.casmi-contest.org/2016/index.shtml) for a set
of 208 MS2 spectra. Later, the MS2 data, used by
[Ruttkies et al. 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9)
for the development of their MS2 + RT score integration framework, was added to
 the database as well.

### Retention Time Datasets

The RankSVM training retention times are _not_ part of this database, but rather
the preference scores predicted by the RankSVM models (see Section 3.2), for the
molecular candidate structures.

## Database Layout

![database_layout](/data/db_layout.png)

### (MS2, RT)-tuples

The table ```challenges``` contains entries corresponding to the datasets used
in our publication (see Section 3.1 and Table 1): CASMI 2016, **positive** and
**negative**; Massbank EA, **EA_positive** and **EA_negative**.

To each challenge the set of spectra can be found in the table ```spectra```. It
contains information about the measured molecule, ionization, spectrum-id,
retention-time etc. Currently, the peak lists, i.e. the actual spectra,
are not stored in the database, but can be found in the [spectra directory](/data/spectra/).

### Candidate Sets

The spectra are associated with their molecular candidates in the table
```candidates_spectra```.

### Representation of the Molecules

All molecules in the database, i.e. candidates and ground truth structures for
the spectra, are stored in the ```molecules``` table. They are indexed (table key) by their
InChI provided by the original creators of the dataset. Furthermore, this table
contains the molecules 2D-InChI, SMILES, molecular formula and references to
exernal databases such as ChemSpider and PubChem (if available).

Across the whole database, the InChI is used as index of the molecules, e.g. for
the fingerprints, candidates, etc., typically refered to as *molecule* column.

#### InChI as Representation for the Molecules

While constructing the database, the structure representation used to calculate
molecular fingerprints or descriptors are InChIs. Even though the database
contains SMILES (canonical and isomeric), those SMILES are calculated from the
InChI representation using [RDKit Python package](http://rdkit.org/docs/api-docs.html):
[MolFromInChI](http://rdkit.org/docs/source/rdkit.Chem.inchi.html#rdkit.Chem.inchi.MolFromInchi)
-> MolToSmiles. This protocol was chosen as some retention time databases only
provide InChI representations.

### Molecular Features

Two molecular feature representations are provided within the database.

#### Molecular Fingerprints (FPS)

Molecular fingerprints describe a molecule by a vector indicating the presence
respectively the count of a set of molecular substructures. The meta description
of the provided fingerprints can be found in table ```fingerprints_meta```. In
total five different fingerprint definitions are in the database (```fingerprints_data```):
- **substructure_count**: Counts of 307 FPS calculated using CDK and
  used for the RankSVM order predictions (see Section 3.2)
- **morgan_***: ECFP and FCFP fingerprints calcualted using RDKit used to study
  the expected similarity for Only MS and MS + RT (not in the paper)
- **iokr_fps__***: Fingerprints used for the positve and negative IOKR models.
  Those have been calculated using CDK and encompass the same FPS definitions as
  in [Dührkop et al. (2019)](https://www.nature.com/articles/s41592-019-0344-8)

The molecular fingerprints are computed for all rows in ```molecules```.

#### Molecular Descriptors

Molecular descriptors calculate meaningful characteristics of molecules, e.g.
LogP, that can be used as input for machine learning prediction tasks. In the
table ```descriptors_meta``` we summarize the provided descriptors and the
corresponding data is available in ```descriptors_data```. However, as the
descriptors where not used in the publication, they are currently only provided
for all molecular structures associated with the CASMI datraset. All descriptors
where calculated using RDKit.

### Spectra Scores

In the database we store the MS2 scores with all spectra typically calculated
with multiple metabolite identification frameworks. Referring to the CASMI
challenge we call the different frameworks *participants* and an overview can
be found in the ```participants``` table. For our publication only 
**MetFrag_2.4.5__*** and **IOKR__696a17f3** are relevant (see Section 3.3).

#### CASMI 2016

For the CASMI dataset, the database contains the MetFrag scores using only
the ```FragmenterScore``` feature (**MetFrag_2.4.5__8afe4a14**) and the IOKR scores
(**IOKR__696a17f3**). Furthermore, it countains combined scores:

(1- D) * ```FragmenterScore``` + D *  ```RetentionTimeScore```

for different values of D. The combination has been directly computed using the
MetFrag software. Check the *description* column in the ```participants``` table 
for details. 

#### Massbank EA

For the Massbank EA dataset, the database contains the MetFrag scores using only
the ```FragmenterScore``` feature (**MetFrag_2.4.5__8afe4a14**) and the IOKR scores
(**IOKR__696a17f3**). The values of MetFrag's ```RetentionTimeScore``` feature are
currently not in the database, but can be found [here](/data/metfrag_RetentionTimeScore_EA)
for each random sample, 50x negative and 100x positive mode (see Section 3.1).
