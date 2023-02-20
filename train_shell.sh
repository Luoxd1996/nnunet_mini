# Train a SwinUNETR on ABOD300 dataset.
CUDA_VISIBLE_DEVICES=0 python -u -m run_training 3d_fullres nnUNetTrainer Task006_ABOD300 all
