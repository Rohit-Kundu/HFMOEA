# Hybrid-FilterGA
Keep the csv file of features in the same directory as these codes and use the following in the command prompt to set the current directory to the code directory.

`cd /d "D:/Hybrid-FilterGA"`

## Requirements
To install the required libraries run the following in the command prompt:

`pip install -r requirements.txt`

## Using the Hybrid-FilterGA Algorithm
For running exhaustive experiments using the whole range of topk1 and topk2 values, run the following code in Command Prompt:

`python Hybrid-FilterGA_main.py --csv_name "SpectEW.csv"`

By default it is assumed that the csv file contains no headers. But if it does, then add the argument `--csv_header yes` in the code above, otherwise an error will be triggered. All csv files must have the class labels in the last column of the file as integer/float values.

## Citation
If you use the codes in this repository or this research helps you in any way, consider citing our paper:

```
@article{kundu2021hybrid,
}
```
