# HFMOEA
This is the official implementation of our paper titled "HFMOEA: A Hybrid Framework for Multi-objective Feature Selection" under peer review in Expert Systems with Applications, Elsevier.

## Requirements
To install the required libraries run the following in the command prompt:

`pip install -r requirements.txt`

## Using the HFMOEA Algorithm
Run the following code in Command Prompt:

`python main.py --root "path_to_csv/" --csv_name "DeepFeatures.csv"`

By default it is assumed that the csv file contains no headers. But if it does, then add the argument `--csv_header "yes"` in the code above, otherwise an error will be triggered. `--root` denotes the folder where the csv file is stored (by default it is `"./"`). All csv files must have the class labels in the last column of the file as integer/float values.
