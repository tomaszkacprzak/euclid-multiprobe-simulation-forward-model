# Deprecated TFRecord pipeline: retained for legacy runs; active pipelines use WebDataset shards.
esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/marcel/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/marcel/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_marcel --log_dir=/pscratch/sd/a/athomsen/run_files/marcel/esub_logs \
    --system=slurm --source_file=../../pipelines/marcel/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../../configs/marcel/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/marcel/tfrecords/fiducial \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_fidu_marcel --log_dir=/pscratch/sd/a/athomsen/run_files/marcel/esub_logs \
    --system=slurm --source_file=../../pipelines/marcel/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

salloc --account=des --constraint=cpu --qos=shared_interactive --time 01:00:00 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=1952