#!/bin/bash

####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

# -- SBATCH --partition=debug --time=01:00:00
# -- SBATCH --nodes=1
# -- SBATCH --cpus-per-task=12 --mem-per-cpu=5000

#SBATCH --partition=batch --time=08:00:00
#SBATCH --nodes=1 --constraint=hsw --exclude=c[579-698] --gres=spindle
#SBATCH --cpus-per-task=12 --mem-per-cpu=5000

#SBATCH --job-name=EA_pos_time

MODE='runtime'
# MODE='debug_runtime'
echo "Mode: $MODE"

# Read script arguments
N_RANDOM_TREES=128
N_SAMPLES=15

ION_MODE=${1}

echo "Number of random spanning trees: $N_RANDOM_TREES"
echo "Dataset EA Massbank (ion_mode=$ION_MODE)"

# Number of jobs to run tree optimization in parallel
N_JOBS=$SLURM_CPUS_PER_TASK
echo "Number of jobs: $N_JOBS"

# Set up file- and directory paths
PROJECTDIR="$SCRATCHHOME/projects/msms_rt_score_integration"
RESULT_DIR="$PROJECTDIR/results/LATEST/EA_Massbank/results__TFG__platt"
EVALSCRIPT="$PROJECTDIR/msmsrt_scorer/experiments/EA_Massbank/runtime__TFG.py"
CASMI_DB_FN="$SCRATCHHOME/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"

# Load the required modules
module purge
module load Python/3.6.6-foss-2018b

# Active the virtual environment
# shellcheck source=/dev/null
source "$PROJECTDIR/venv/bin/activate"

# Sleep for some time to prevent conflicts when creating
# directories within the evaluations scripts.
sleep $(( ( ${RANDOM} % 15 ) + 1 ))

# Create temporary output directory for results on local disk of node
BASE_ODIR="/tmp/$SLURM_JOB_ID"
mkdir "$BASE_ODIR" || exit 1

# Change to output directory
cd "$BASE_ODIR" || exit 1

# Set up trap to remove my results on exit from the local disk
trap "rm -rf $BASE_ODIR; exit" TERM EXIT

# Run the evaluation scripts
if [ $MODE = "runtime" ]
then
  srun python "$EVALSCRIPT" \
      --mode="$MODE" \
      --D_value_grid 0.001 0.005 0.01 0.05 0.1 0.15 0.25 0.35 0.5 \
      --database_fn="$CASMI_DB_FN" \
      --n_jobs="$N_JOBS" \
      --base_odir="$BASE_ODIR" \
      --n_samples="$N_SAMPLES" \
      --n_random_trees="$N_RANDOM_TREES" \
      --ion_mode="$ION_MODE" \
      --max_n_ms2_grid 15 30 45 60 75
elif [ $MODE = "debug_runtime" ]
then
  srun python "$EVALSCRIPT" \
      --mode="$MODE" \
      --D_value_grid 0.01 0.1 0.5 \
      --database_fn="$CASMI_DB_FN" \
      --n_jobs="$N_JOBS" \
      --base_odir="$BASE_ODIR" \
      --n_samples=3 \
      --n_random_trees=4 \
      --ion_mode="$ION_MODE" \
      --max_n_ms2_grid 15 45
else
  echo "Invalid mode: $MODE"
  exit 1
fi

# Tar results and copy them into my work-directory
# we are in the output directory
TAR_FN="results_"$MODE"_$SLURM_JOB_ID.tar"

tar -cf "$TAR_FN" $MODE
mv "$TAR_FN" "$RESULT_DIR"
tar -xf "$RESULT_DIR/$TAR_FN" -C "$RESULT_DIR"
