# debug

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v15/debug/extended_debug.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/webdatasets/grid \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --tasks="0" \
    --job_name=wds_grid_v15_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 --no_derivatives \
    --config=../../configs/v15/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=rerun_missing --n_jobs=1000 --max_njobs=1000 --tasks="0>1000" --keep_submit_files \
    --job_name=wds_fidu_v15 --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 --no_derivatives \
    --config=../../configs/v15/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/webdatasets/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=merge --n_jobs=1000 --max_njobs=1000 --tasks="0>1000" --keep_submit_files \
    --job_name=wds_fidu_v15 --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"