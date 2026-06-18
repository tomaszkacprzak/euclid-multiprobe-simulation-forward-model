#! /bin/bash
#
#SBATCH --output=/global/u2/a/athomsen/multiprobe-simulation-forward-model/esub_logs/wds_fidu_test_main_index%a.o
#SBATCH --error=/global/u2/a/athomsen/multiprobe-simulation-forward-model/esub_logs/wds_fidu_test_main_index%a.e
#SBATCH --job-name=wds_fidu_test_main
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --array=1-1
#SBATCH --nodes=1
#SBATCH --time=0:30:00

sleep 10
