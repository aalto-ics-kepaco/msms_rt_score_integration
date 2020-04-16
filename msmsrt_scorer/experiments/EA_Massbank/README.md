## Install RDKit

**Only if you want to run the MetFrag 2.2 evaluation again**

To run the MetFrag 2.2 evaluation on the EA (Massbank) dataset, you additionally need to install 
RDKit **with InChI** support. That is because the MetFrag retention time scores, are not stored
in the SQLite DB yet. The instructions how to install RDKit can be found [here](http://www.rdkit.org/docs/Install.html#linux-and-os-x).
The user maintained repository of Arch Linux contains a [build-file for RDKit](https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=rdkit&id=1a790a8756bee5e39a5efc751c3ed5b4af30ea49)
giving you additional guidance how to install the package.