# Deprecated TFRecord pipeline: retained for legacy runs; active pipelines use WebDataset shards.
#!/bin/bash
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cl_merge
#SBATCH --output="./logs/v8/cl_merge%j.log"

# SIMSET="fiducial"
SIMSET="grid"

srun --cpu-bind=cores \
    python ../../../msfm/apps/perlmutter/merge_${SIMSET}_cls.py \
    --dir_out="/pscratch/sd/a/athomsen/DESY3/v8/linear_bias/webdatasets/${SIMSET}" \
    --config="/global/u2/a/athomsen/multiprobe-simulation-forward-model/configs/v8/linear_bias.yaml" \
    --file_suffix=""
