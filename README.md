# HFMOEA
This is the official implementation of our paper titled "HFMOEA: A Hybrid Framework for Multi-objective Feature Selection" under peer review in Journal of Computational Design and Engineering, Oxford.

## Requirements
To install the required libraries run the following in the command prompt:

`pip install -r requirements.txt`

## Using the HFMOEA Algorithm
Run the following code in Command Prompt:

`python main.py --path path/to/file/csv_name.csv`

By default it is assumed that the csv file contains no headers. But if it does, then add the argument `--csv_header True` in the code above, otherwise an error will be triggered. All csv files must have the class labels in the last column of the file as integer/float values.

Other available arguments are listed as follows:
- `popsize`: Population Size (Note: must be equal to or more than 10, since 10 filter methods are used to initialize part of the population- refer to our paper for more details)
- `generations`: Number of generations for the HFMOEA algorithm
- `mutation`: Percentage of mutation
- `topk`: "topk" number of features (Please refer to our paper for more details)
- `save_fig`: Whether the plots need to be saved or not
