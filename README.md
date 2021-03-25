# cgts: Correlation graph analytics for time series 


## Install anaconda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

For more information, see
https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

Then, configure environment paths.
MacOS: .zprofile
Linux: .bashrc

## Install libraries

conda install --file requirements.txt

Or run the following if you prefer manual installations
``
conda install -c conda-forge ipycytoscape
conda install pandas
conda install scipy
conda install -c anaconda networkx 
conda install matplotlib
``

## Launch 

:cgts$ jupyter notebook

open ``main.ipynb``


## Utils
use `dd` to delete rows




License :copyright:
===
The ``cgts`` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See http://www.gnu.org/licenses/gpl.html.