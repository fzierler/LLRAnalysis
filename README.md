# LLRAnalysis
This repository used for the analysis of the results of LLR method for SU(3) lattice gauge theory using a modified version of Hirep (https://github.com/dave452/Hirep-LLR-SU).
It can read the output file from HiRep and analyses it, this is then saved into CSV files.
These CSV files can be analysed and the results plotted.
Analysis used for the results of the paper: Lucini, B., Mason, D., Piai, M., Rinaldi, E., & Vadacchino, D. (2023). First-order phase transitions in Yang-Mills theories and the density of state method. arXiv preprint arXiv:2305.07463. 
-------------------------------------------------------------------------
## Creating Conda environment
To create the conda environment in terminal with conda installed use: <br/>
conda env create -f environment.yml

Then to activate the conda environment use:<br/>
conda activate llr
 
Once the enivroment is activated, install the analysis code with:<br/>
python setup.py install

-------------------------------------------------------------------------
To reproduce results of the paper, install this python package, download the data release and run the jupyter notebooks within the data release (reference).
