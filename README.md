# Hybrid_MOEA
This is the official implementation of our paper titled "" under peer review in .

## Requirements
To install the required libraries run the following in the command prompt:

`pip install -r requirements.txt`

## Using the Hybrid-FilterGA Algorithm
For running exhaustive experiments using the whole range of topk1 and topk2 values, run the following code in Command Prompt:

`python main.py --root "csvUCI/" --csv_name "SpectEW.csv"`

By default it is assumed that the csv file contains no headers. But if it does, then add the argument `--csv_header "yes"` in the code above, otherwise an error will be triggered. `--root` denotes the folder where the csv file is stored (by default it is `"./"`). All csv files must have the class labels in the last column of the file as integer/float values.
