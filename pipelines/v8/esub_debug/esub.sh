# on Euler
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 \
    --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1000 --tasks="0>1000" \
    --job_name=wds_fidu_test --system=slurm --test --keep_submit_files

#! /bin/bash
#
#SBATCH --output=/cluster/home/athomsen/dlss/repos/multiprobe-simulation-forward-model/esub_logs/wds_fidu_test_main_index%a.o
#SBATCH --error=/cluster/home/athomsen/dlss/repos/multiprobe-simulation-forward-model/esub_logs/wds_fidu_test_main_index%a.e
#SBATCH --job-name=wds_fidu_test_main
#SBATCH --array=1-1000%50
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=4096
#SBATCH --cpus-per-task=8

srun bash
python -m esub.submit --job_name=wds_fidu_test --tasks='0>1000' --source_file=source_esub.sh --main_memory=4096 --main_time=4 --main_scratch=4096 --function=main --executable=/cluster/home/athomsen/dlss/repos/multiprobe-simulation-forward-model/msfm/apps/run_fiducial_preprocessing.py --n_jobs=1000 --log_dir=/cluster/home/athomsen/dlss/repos/multiprobe-simulation-forward-model/esub_logs --system=slurm --main_name=main --batchsize=100000 --max_njobs=50 --main_n_cores=8 --main_gpu=0 --main_gpu_memory=1000 --esub_verbosity=3 --main_mode=jobarray --mode=jobarray --n_files=1000 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/linear_bias/webdatasets/fiducial --cosmogrid_version=1.1

# on Perlmutter
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1000 --tasks="0>1000" --main_n_cores=1 \
    --job_name=wds_fidu_test --system=slurm --per_node_accounting --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --test --keep_submit_files --debug

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --tasks="0>3" --main_n_cores=1 --main_time=0.1 \
    --job_name=wds_fidu_test --system=slurm --per_node_accounting --source_file="pipelines/v8/esub_debug/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --debug

source pipelines/v8/esub_debug/perlmutter_setup.sh

# test on one node
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=128 --tasks="0>128" --main_n_cores=1 \
    --job_name=wds_fidu --system=slurm --per_node_accounting --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=128 --tasks="0>128" --main_n_cores=1 \
    --job_name=wds_fidu_3 --system=slurm --per_node_accounting --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files --test

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=128 --tasks="0>128" --main_n_cores=1 --main_time=0.1 \
    --job_name=wds_fidu_2_debug --system=slurm --per_node_accounting --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --debug

# salloc
salloc --qos shared_interactive --time 00:30:00 --constraint cpu --account=des --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
source pipelines/v8/perlmutter_setup.sh

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug/fiducial \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --tasks="0" --main_n_cores=4 --main_time=0.1 \
    --job_name=wds_fidu_interactive --system=slurm --per_node_accounting --source_file="pipelines/v8/esub_debug/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files --debug

# euler
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 \
    --dir_out=/cluster/work/refregier/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_njobs=50 --tasks="0" \
    --job_name=wds_fidu_v8 --system=slurm

# with more cores per task
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1000 --tasks="0>1000" --per_node_accounting \
    --main_n_cores=4 --main_memory=4000 --main_time=1 \
    --job_name=wds_fidu_test --system=slurm --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files --test

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=32 --tasks="0>32" --per_node_accounting \
    --main_n_cores=4 --main_memory=4000 --main_time=4 \
    --job_name=wds_fidu_test --system=slurm --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files

# grid

# submission
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9 \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=64 --max_nnodes=100 --tasks="0>64" --per_node_accounting \
    --main_n_cores=4 --main_memory=4000 --main_time=8 \
    --job_name=wds_grid --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files

# debug
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v8/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=rerun_missing --n_jobs=64 --max_nnodes=100 --tasks="0>64" --per_node_accounting \
    --main_n_cores=4 --main_memory=4000 --main_time=0.5 \
    --job_name=wds_grid --system=slurm --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files

# interactive
source pipelines/v9/perlmutter_setup.sh
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9 \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_nnodes=100 --tasks="0" --per_node_accounting \
    --main_n_cores=4 --main_memory=4000 --main_time=4 \
    --job_name=wds_grid --system=slurm --source_file="pipelines/v8/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files --debug

# partial run
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=64 --max_nnodes=100 --tasks="0>64" --per_node_accounting \
    --job_name=wds_grid_v9 --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=64 --max_nnodes=100 --tasks="0>64" --per_node_accounting \
    --job_name=wds_fidu_v9 --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=regular" \
    --keep_submit_files

# debug
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --max_nnodes=100 --tasks="0" --per_node_accounting \
    --main_n_cores=8 --main_time=0.5 \
    --job_name=wds_fidu_v9_serial --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --debug --verbosity=debug

taskset --cpu-list 1,2,3,4
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=16 --max_nnodes=100 --tasks="0>16" --per_node_accounting \
    --main_n_cores=8 --main_time=0.5 \
    --job_name=wds_fidu_v9 --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --debug --verbosity=debug

# from community
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/cfs \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=32 --max_nnodes=100 --tasks="0>32" --per_node_accounting \
    --main_time=0.5 \
    --job_name=wds_grid_cfs --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --verbosity=debug

esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/cfs \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --max_nnodes=100 --tasks="0" --per_node_accounting \
    --main_time=0.5 \
    --job_name=wds_grid_cfs --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --verbosity=debug

esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/cfs \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=4 --max_nnodes=100 --tasks="0>4" --per_node_accounting \
    --main_time=0.5 --main_n_cores=8 \
    --job_name=wds_grid_four --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --verbosity=debug

# from scratch
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/pscratch/sd/a/athomsen/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/scratch \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=32 --max_nnodes=100 --tasks="0>32" --per_node_accounting \
    --main_time=0.5 \
    --job_name=wds_grid_scratch --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --verbosity=debug

esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/pscratch/sd/a/athomsen/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/scratch \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=32 --max_nnodes=100 --tasks="0>32" --per_node_accounting \
    --main_time=0.5 \
    --job_name=wds_grid_scratch_interactive --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug" \
    --keep_submit_files --verbosity=debug

chmod -R g+rwx /pscratch/sd/j/jbucko/ArneDestroyingMyScratch

cat wds_grid_cfs_main_index1.o
cat wds_grid_scratch_main_index1.o

# shared queue
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/cfs \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=100 --tasks="0>100" \
    --job_name=wds_grid_shared --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared" \
    --keep_submit_files

# debug

# fiducial
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=31 --max_njobs=1000 --tasks="1>32" \
    --job_name=wds_fidu_shared_debug --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared"

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=31 --max_njobs=1000 --tasks="1>32" \
    --job_name=wds_fidu_shared_debug --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared"

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="1" \
    --job_name=wds_fidu_shared_debug --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared" --debug

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="1" \
    --job_name=wds_fidu_shared_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared"

# grid
esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/grid/cfs \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --tasks="0" \
    --main_n_cores=4 --main_time=1 \
    --job_name=wds_grid_shared_debug --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared" --keep_submit_files

cat wds_grid_shared_debug_main_index1.o
cat wds_fidu_shared_debug_main_index1.o

# esub doesn't stop
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/debug \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=wds_fidu_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" --debug

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/debug \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=wds_fidu_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file="pipelines/v9/perlmutter_setup.sh" \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared" --debug

esub msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/debug \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --tasks="0" \
    --main_n_cores=4 --main_time=1 \
    --job_name=wds_grid_shared_debug --system=slurm --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared" \
    --keep_submit_files --debug

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/DESY3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1000 --max_njobs=1000 --tasks="0>1000" \
    --job_name=wds_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --test --keep_submit_files

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=10 --max_njobs=1000 --tasks="0>10" --keep_submit_files \
    --job_name=wds_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=../configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1000 --max_njobs=1000 --tasks="0>1000" --keep_submit_files \
    --job_name=wds_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=../pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" --test

esub ../msfm/apps/run_grid_preprocessing.py \
    --n_files=2500 \
    --config=../configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=wds_grid_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=../pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" --test

esub ../msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=../configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=10 --max_njobs=1000 --tasks="0>10" --keep_submit_files \
    --job_name=wds_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=../pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# per node, --qos=regular
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias.yaml \
    --dir_in=/pscratch/sd/a/athomsen/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=10 --max_nnodes=100 --tasks="0>10" --per_node_accounting \
    --job_name=wds_fidu_node --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --main_time=0.5 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=debug"

# no smoothing
esub msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=configs/v9/linear_bias_no_smoothing.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias_no_smoothing/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=wds_fidu_v9_no_smoothing --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=../configs/v9/linear_bias.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=10 --max_njobs=1000 --tasks="0>10" --keep_submit_files \
    --job_name=wds_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=../pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
