#!/bin/bash

# Stop the execution if any error occurs
set -e

if [ "$#" -eq 2 ]; then
	BOARD=$1
	MODEL_NAME=$2
else
	echo "Provide a valid BOARD and MODEL_NAME as arguments."
	exit 1
fi

# Copy the arch.json file containing the fingerprint of the Ultra96v2
sudo mkdir -p /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/
sudo cp arch.json /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/

compile() {
    vai_c_tensorflow2 \
        --model ./model/${MODEL_NAME}.h5 \
        --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/${BOARD}/arch.json \
        --output_dir ./xmodel \
        --net_name HippoScan_${BOARD}
}

mkdir -p xmodel/
mkdir -p logs/compile/
# Compile the model (Redirect the error output to stdout and save in the log file)
compile 2>&1 | tee logs/compile/HippoScan_$BOARD.log