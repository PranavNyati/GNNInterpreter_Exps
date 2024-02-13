#!/bin/bash
#SBATCH --job-name=gnni_GCN_mutagenicity_train # Job name
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --mem=4gb # Job memory request
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --gpus-per-node=2 # Number of GPU
#SBATCH --partition=gpupart_p100 # Time limit hrs:min:sec
#SBATCH --time=12:05:00 # Time limit hrs:min:sec
#SBATCH --error=/home/du0/20CS30037/BTP_Experiments/Pranav_GNNInter/GNNInterpreter_Exps/GNNInterpreter/server_run_logs/gnni_GCN_mutagenicity_train_%J.err_  # For error logs
#SBATCH --output=/home/du0/20CS30037/BTP_Experiments/Pranav_GNNInter/GNNInterpreter_Exps/GNNInterpreter/server_run_logs/gnni_GCN_mutagenicity_train_hd_64_epochs_128_%J.out_  # Standard output log
#SBATCH --mail-user=pranavnyati26@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job



source /home/du0/20CS30037/miniconda3/bin/activate gnn_inter
cd /home/du0/20CS30037/BTP_Experiments/Pranav_GNNInter/GNNInterpreter_Exps/GNNInterpreter
srun python3 train_base_model.py  > server_run_logs/gnni_GCN_mutagenicity_train_hd_64_epochs_128.log
# srun python3 vrrw.py --dataset proteins
