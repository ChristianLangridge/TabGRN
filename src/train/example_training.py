
# this file describes what running a paths setup when working on a training run using a multi-GPU HPC. 
import torch.distributed as dist
from spatialmt.config import Paths, setup_output_dirs, validate_raw_inputs

# Only rank 0 touches the filesystem
if dist.get_rank() == 0:
    setup_output_dirs()
    validate_raw_inputs()

# All ranks wait here until rank 0 finishes
dist.barrier()

# Now all ranks can safely read data
dataset = MyDataset(Paths.processed_tpm)