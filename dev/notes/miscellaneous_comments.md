- Path to the full sky CosmoGrid maps with DES Y3 redshift bins
  - on the SAN `/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3`
  - transfer to Euler scratch `sbatch -n 4 --cpus-per-task=4 --time=4:00:00 --mem-per-cpu=1024 --wrap="rsync -ahv --include='*/' --include='projected_probes_maps_nobaryons512.h5' --exclude='*' athomsen@login.phys.ethz.ch:/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3/grid /cluster/scratch/athomsen/CosmoGrid"`
  - at NERSC `/global/cfs/cdirs/des/cosmogrid/DESY3`
- Start jupyter on Euler `./start_jupyter_nb.sh -u athomsen -l -W 04:00 --n 16 -m 2048 -s new -w /cluster/home/athomsen/des/repos/multiprobe-simulation-forward-model/notebooks`
- Run on a set number of cores (for debugging in an interactive session): `taskset --cpu-list 0-16`  
- Euler interactive session `srun -n 1 --cpus-per-task=8 --time=1:00:00 --mem-per-cpu=2048 --pty bash`
- Euler `work` to SAN for WebDataset shards `sbatch -n 4 --cpus-per-task=4 --time=4:00:00 --mem-per-cpu=1024 --wrap="rsync -ahv --include='*/' --include='*.tar' --exclude='*' /cluster/work/refregier/athomsen/CosmoGrid/DESY3/v4/large_scales/webdatasets athomsen@login.phys.ethz.ch:/home/ipa/refreg/experiments/athomsen/CosmoGrid/DESY3/v4/large_scales"` (or Globus).
- V1.1 update: `sbatch -n 4 --cpus-per-task=4 --time=4:00:00 --mem-per-cpu=1024 --wrap="rsync -ahv --include='*/' --include='projected_probes_maps_v11dmo.h5' --include='metainfo_perms.npy' --exclude='*' athomsen@login.phys.ethz.ch:/home/ipa/refreg/data/data_products/CosmoGrid/processed/v11desy3 /cluster/scratch/athomsen/CosmoGrid/raw"`

