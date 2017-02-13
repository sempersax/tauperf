# Current Workflow
## Download sample 

```bash
rucio download user.qbuat.mc15_13TeV.361108.PowPy8EvtGen_AZNLOCTEQ6L1_Ztautau.recon.ESD.e3601_s2650_s2183_r7823.tauid.v8_OUT
```

## Select taus and create 1 tree per decay mode

```bash
tau-tree-selector /path/to/dataset/
```

## merge the root files

```bash
hadd output_selected.root /path/to/dataset/user.*selected.root
```

## make the h5 file

```bash
root2hdf5 output_selected.root
```

## create the images

```bash
tau-image /path/to/h5/file.h5 --tau-type 1p0n
tau-image /path/to/h5/file.h5 --tau-type 1p1n
tau-image /path/to/h5/file.h5 --tau-type 1p2n
tau-image /path/to/h5/file.h5 --tau-type 3p0n
tau-image /path/to/h5/file.h5 --tau-type 3p0n
```
## fit and evaluate!
For example:

```bash
python fitter_dense.py --overwrite
```
