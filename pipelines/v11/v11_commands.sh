# Deprecated TFRecord pipeline: retained for legacy runs; active pipelines use WebDataset shards.
# debug
esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v11/debug/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/debug \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=2500 --max_njobs=1000 --tasks="0" \
    --job_name=wds_grid_v11_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --file_suffix="extended" \
    --debug

esub ../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../configs/v11/power_law.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/power_law/webdatasets/debug/fiducial \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1000 --max_njobs=1000 --tasks="0" \
    --job_name=wds_fidu_v11_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --file_suffix="power_law" \
    --debug

esub ../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../configs/v11/power_law.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/power_law/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=100 --max_njobs=1000 --tasks="0" \
    --job_name=wds_fidu_v11 --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v11/debug/extended_hard_cut.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/extended_hard_cut/webdatasets/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=wds_grid_v11 --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# production
esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v11/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/extended/webdatasets/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" \
    --job_name=wds_grid_v11 --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v11/extended_hard_cut.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/extended_hard_cut/webdatasets/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" \
    --job_name=wds_grid_v11 --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../configs/v11/power_law.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v11/power_law/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=100 --max_njobs=1000 --tasks="0>100" \
    --job_name=wds_fidu_v11 --log_dir=/pscratch/sd/a/athomsen/run_files/v11/esub_logs \
    --system=slurm --source_file=../pipelines/v11/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
