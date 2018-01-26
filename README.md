# CapsNet
Implementation of the Capsule Network architecture with tensorflow

## How to run on school computers

```sh
$ source capsenv.sh
```
Loads requred modules for tensorflow GPu support, created a conda enviroment and activates is. Jupyter notebook can be now started with the normal start command.
```sh
$ jupyter notebook
```
New packege requrements updated by:
```sh
$ conda env export > environment.yml
```
Run notebook in bash and save result to new notebook:
```sh
$ jupyter nbconvert --Application.log_level=10 --ExecutePreprocessor.timeout=-1 --to notebook --execute notebook_name.ipynb
```
