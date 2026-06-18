# debug

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v13/debug/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v13/extended/webdatasets/grid \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=wds_grid_v13 --log_dir=/pscratch/sd/a/athomsen/run_files/v13/esub_logs \
    --system=slurm --source_file=../../pipelines/v13/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../../configs/v13/debug/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v13/extended/webdatasets/fiducial \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=wds_fidu_v13 --log_dir=/pscratch/sd/a/athomsen/run_files/v13/esub_logs \
    --system=slurm --source_file=../../pipelines/v13/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
