# nnunet_mini

# Usage
* 1. use the original nnUNet to do data processing and generate training planning.
* 2. change the path in configuration.py to your own.
* 3. use your own network to replace the original Generic_UNet in "nnUNetTrainer.py", line 275, function "initialize_network()".
* 4. train your own network using the "train_shell.py".