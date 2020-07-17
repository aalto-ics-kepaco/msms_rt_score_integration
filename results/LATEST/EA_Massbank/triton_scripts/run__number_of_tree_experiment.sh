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

# -- SBATCH --partition=debug --time=01:00:00 --nodes=1
# -- SBATCH --cpus-per-task=8 --mem-per-cpu=4000

# n_random_trees = 1
#SBATCH --partition=batch --time=01:00:00 --nodes=1

# n_random_trees = 2
# -- SBATCH --partition=batch --time=01:30:00 --nodes=1

# n_random_trees = 4
# -- SBATCH --partition=batch --time=02:00:00 --nodes=1

# n_random_trees = 8
# -- SBATCH --partition=batch --time=03:00:00 --nodes=1

# n_random_trees = 16
# -- SBATCH --partition=batch --time=05:00:00 --nodes=1

# n_random_trees = 32
# -- SBATCH --partition=batch --time=08:00:00 --nodes=1

# n_random_trees = 64
# -- SBATCH --partition=batch --time=12:00:00 --nodes=1

# n_random_trees = 128
# -- SBATCH --partition=batch --time=18:00:00 --nodes=1

# n_random_trees = 256
#SBATCH --partition=batch --time=36:00:00 --nodes=1

# Positive
# -- SBATCH --cpus-per-task=36 --mem-per-cpu=4000

# Negative
#SBATCH --cpus-per-task=20 --mem-per-cpu=3000

#SBATCH --job-name=EA_pos_001_max

# MODE='debug_application'
MODE='application'
echo "Mode: $MODE"

# Read script arguments
TREE_METHOD="random"
MAKE_ORDER_PROB="sigmoid"
PARAM_SELECTION_MEASURE="topk_auc"

ION_MODE=${1}
if [ $ION_MODE = "positive" ]
then
  MAX_N_MS2=100
  N_SAMPLES=100
elif [ $ION_MODE = "negative" ]
then
  MAX_N_MS2=65
  N_SAMPLES=50
else
  echo "Invalid ionization mode: $ION_MODE"
  exit 1
fi
N_RANDOM_TREES=${2}
MTYPE=${3}


echo "tree-method: $TREE_METHOD (with n_trees=$N_RANDOM_TREES)"
echo "Function for preference score conversion: $MAKE_ORDER_PROB"
echo "Dataset EA (ion_mode=$ION_MODE) with max_n_ms2=$MAX_N_MS2"
echo "Parameter selection measure: $PARAM_SELECTION_MEASURE"
echo "Margin type: $MTYPE"

# Number of jobs to run tree optimization in parallel
N_JOBS=$SLURM_CPUS_PER_TASK
echo "Number of jobs: $N_JOBS"

# Set up file- and directory paths
PROJECTDIR="$SCRATCHHOME/projects/msms_rt_score_integration"
RESULT_DIR="$PROJECTDIR/results/LATEST/EA_Massbank/results__TFG__platt"
EVALSCRIPT="$PROJECTDIR/msmsrt_scorer/experiments/EA_Massbank/eval__TFG.py"
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
if [ $MODE = "application" ]
then
  srun python "$EVALSCRIPT" \
      --mode="$MODE" \
      --param_selection_measure="$PARAM_SELECTION_MEASURE" \
      --D_value_grid 0.001 0.005 0.01 0.05 0.1 0.15 0.25 0.35 0.5 \
      --order_prob_k_grid platt \
      --database_fn="$CASMI_DB_FN" --n_jobs="$N_JOBS" \
      --base_odir="$BASE_ODIR" --n_samples="$N_SAMPLES" --n_random_trees="$N_RANDOM_TREES" --ion_mode="$ION_MODE" \
      --max_n_ms2="$MAX_N_MS2" \
      --make_order_prob="$MAKE_ORDER_PROB" \
      --margin_type="$MTYPE"
elif [ $MODE = "debug_application" ]
then
  srun python "$EVALSCRIPT" \
      --mode="$MODE" \
      --param_selection_measure="$PARAM_SELECTION_MEASURE" \
      --D_value_grid 0.01 0.1 0.5 \
      --order_prob_k_grid platt \
      --database_fn="$CASMI_DB_FN" --n_jobs="$N_JOBS" \
      --base_odir="$BASE_ODIR" --n_samples=3 --n_random_trees=4 --ion_mode="$ION_MODE" \
      --max_n_ms2="$MAX_N_MS2" \
      --make_order_prob="$MAKE_ORDER_PROB" \
      --margin_type="$MTYPE"
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
