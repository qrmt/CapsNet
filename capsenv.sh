#!/bin/bash


echo "Loading modules."

module load anaconda3 CUDA cuDNN

echo "Done."

ENV="capsenv"

echo "Starting conda enviroment '$ENV'"

source activate $ENV
if [ $? -ne 0 ]; then
	echo "Conda env '$ENV' doesn't exist."
	echo "Creating enviroment '$ENV':"
	conda env create --name capsenv --file environment.yml
	source activate $ENV
fi


