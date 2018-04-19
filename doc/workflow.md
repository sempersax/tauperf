# Current Workflow
## Download sample 

```bash
rucio download user.qbuat.mc16_13TeV.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight.recon.ESD.e5468_s3170_r9466.tauid.v2_OUT
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
For example
```bash
python fitter_dense_multi.py --overwrite
```
