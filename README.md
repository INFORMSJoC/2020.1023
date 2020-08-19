[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# CacheTest

This project is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are associated with the paper [This is a Template](https://doi.org/10.1287/ijoc.2019.0934) by T. Ralphs. 

## Version

The version used in the paper is

[![Release](https://img.shields.io/github/v/release/INFORMSJoC/Template?sort=semver)](https://github.com/INFORMSJoC/Template/releases)

## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2019.0934) and the software, using the following DOI.

[![DOI](https://zenodo.org/badge/285853815.svg)](https://zenodo.org/badge/latestdoi/285853815)

## Description

The goal of this software is to demonstrate the effect of cache optimization.

## Building

In Linux, to build the version that multiplies all elements of a vector by a
constant (used to obtain the results in [Figure 1](results/mult-test.png) in the
paper), stepping K elements at a time, execute the following commands.

```
make mult
```

Alternatively, to build the version that sums the elements of a vector (used
to obtain the results [Figure 2](results/sum-test.png) in the paper), stepping K
elements at a time, do the following.

```
make clean
make sum
```

Be sure to make clean before building a different version of the code.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
