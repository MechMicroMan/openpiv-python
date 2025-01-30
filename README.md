# OpenPIV XL

The authors aknowledge the authors of [OpenPIV](https://github.com/OpenPIV/openpiv-python), from which this package is derived. OpenPIV should be cited with this [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4409178).

This package is designed to perform DIC on large images (> 10,000 x 10,000 px) using chunking and GPU acceleration.

Using the GPU, DIC correlation from a sub-window of 2048x2048 to 16x16 with no overlap on a (16000x16000) px image takes around a minute.

## Installing

Download the package and run

`pip install -e .`

OR

`pip install git+https://github.com/MechMicroMan/openpiv-python.git`


## License and copyright

This package is licenses under the GNU General Public License v3.0

Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

