# Weisfeiler-Lehman Filtration Kernel

## Dependencies:
- tqdm==4.28.1
- networkx==2.4
- numpy==1.19.1
- scikit_learn==0.24.2

## Run the code:
Example: python3 Main.py EGO-2

NOTE: files in data folder need to be unzipped first


Additional arguments are as follows:
positional arguments:
  db                    	Dataset name

optional arguments:
  -h, --help            	show this help message and exit
  --h H [H ...]         	List of WL depths values
  --k K [K ...]         	List of filtration lengths
  --gamma GAMMA [GAMMA ...]	List of gammas            
  --c C [C ...]         	List of Cs (SVM parameter)
  --jobs JOBS           	Number of parallel processes

