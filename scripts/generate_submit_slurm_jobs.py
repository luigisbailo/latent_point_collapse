import os
import argparse
import shutil

template_job = r"""#!/bin/bash -l
#SBATCH --gres=gpu:a100:${n_gpus}
#SBATCH --time=${hours}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${n_cpus}
#SBATCH --export=NONE
#SBATCH --output=${output_file}
#SBATCH --job-name=${job_name}

unset SLURM_EXPORT_ENV
module load python
module load cuda
conda activate lpc_ib

while :
do
MASTER_PORT=$(shuf -i 20000-65000 -n 1)
if ! lsof -i:$MASTER_PORT -t >/dev/null; then
break
fi
done


echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $SLURM_LAUNCH_NODE_IPADDR"


srun -N "$SLURM_JOB_NUM_NODES" --cpu-bind=verbose \
  /bin/bash -c "torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc-per-node=\$SLURM_GPUS_ON_NODE --master-addr=\$SLURM_LAUNCH_NODE_IPADDR --master-port=\"$MASTER_PORT\" --start-method=forkserver --node-rank=\$SLURM_NODEID  ${model_script} --config  ${config}  --lr ${lr} --encoding-metrics True --store-penultimate True --results-dir ${results_dir}/${id_name}/${k_dir}  --dataset-dir ${dataset_dir} --sample ${i_lr}"
"""

# Generate job scripts and submit them
def submit_slurm_jobs(template, config, id_name, dataset_dir, results_dir, output_dir, start_lr, factor_lr, steps_lr, epochs, logging, hours, n_gpus, n_cpus):
    
    create_replace_directory(f"{results_dir}/{id_name}")
    create_replace_directory(f"{output_dir}/{id_name}")
    for k_dir in [1]:
        os.makedirs(f"{results_dir}/{id_name}/{k_dir}", exist_ok=True)
        os.makedirs(f"{output_dir}/{id_name}/{k_dir}", exist_ok=True)        
        shutil.copy2(config, f"{results_dir}/{id_name}/{k_dir}")
 
        script_path = os.path.dirname(os.path.abspath(__file__))
        lr = start_lr
        for i_lr in range(steps_lr + 1):
            script_parameters = [
                ("ib", f"{output_dir}/{id_name}/{k_dir}/ib_{i_lr}.out", f"{script_path}/main.py --model-name ib --loss-encoding True --dropout-penultimate False"),
                ("ib_wide", f"{output_dir}/{id_name}/{k_dir}/ib_wide_{i_lr}.out", f"{script_path}/main.py  --model-name ib --loss-encoding True --penultimate-nodes wide --dropout-penultimate False"),
                ("ib_narrow", f"{output_dir}/{id_name}/{k_dir}/ib_narrow_{i_lr}.out", f"{script_path}/main.py  --model-name ib --loss-encoding True --penultimate-nodes narrow --dropout-penultimate False"),
                ("no_pen", f"{output_dir}/{id_name}/{k_dir}/no_pen_{i_lr}.out", f"{script_path}/main.py  --model-name no_pen --loss-encoding False --dropout-penultimate False"),
                ("no_pen_dropout", f"{output_dir}/{id_name}/{k_dir}/no_pen_dropout_{i_lr}.out", f"{script_path}/main.py  --model-name no_pen --loss-encoding False --dropout-penultimate True"),
                ("lin_pen", f"{output_dir}/{id_name}/{k_dir}/lin_pen_{i_lr}.out", f"{script_path}/main.py  --model-name lin_pen --loss-encoding False --dropout-penultimate False"),
                ("lin_pen_dropout", f"{output_dir}/{id_name}/{k_dir}/lin_pen_dropout_{i_lr}.out", f"{script_path}/main.py  --model-name lin_pen --loss-encoding False --dropout-penultimate True"),
                ("nonlin_pen", f"{output_dir}/{id_name}/{k_dir}/nonlin_pen_{i_lr}.out", f"{script_path}/main.py  --model-name nonlin_pen --loss-encoding False --dropout-penultimate False"),
            ]

            for model_name, output_file, model_script in script_parameters:
                job_script = template.replace("${n_gpus}", str(n_gpus)) \
                                     .replace("${n_cpus}", str(n_cpus)) \
                                     .replace("${hours}", str(hours)) \
                                     .replace("${output_file}", output_file) \
                                     .replace("${job_name}", f"{id_name}_{k_dir}_{i_lr}_{model_name}") \
                                     .replace("${config}", config) \
                                     .replace("${lr}", str(lr)) \
                                     .replace("${results_dir}", results_dir) \
                                     .replace("${id_name}", id_name) \
                                     .replace("${dataset_dir}", dataset_dir) \
                                     .replace("${k_dir}", str(k_dir)) \
                                     .replace("${i_lr}", str(i_lr)) \
                                     .replace("${model_script}", model_script)
                script_filename = f"run_job_{model_name}_{i_lr}_{k_dir}.sh"

                with open(script_filename, 'w') as f:
                    f.write(job_script)

                os.system(f"sbatch {script_filename}")
                os.remove(script_filename)

            lr *= factor_lr
            
def create_replace_directory(dir_path):
        
    if os.path.exists(dir_path):
        prompt1 = input(f"The directory {dir_path} already exists. Do you want to remove it and create a new one? (Y/N): ")
        if prompt1.lower() == 'y':
            prompt2 = input(f"Are you sure you want to remove the directory {dir_path}? This action cannot be undone. (Y/N): ")
            if prompt2.lower() == 'y':
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                print(f"Removed and recreated the directory: {dir_path}")
            else:
                print(f"Action canceled. Using the existing directory: {dir_path}")
        else:
            print(f"Using the existing directory: {dir_path}")
    else:
        os.makedirs(dir_path)
        print(f"Created the directory: {dir_path}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and submit SLURM job scripts.")
    parser.add_argument('--hours', type=str, required=True, help="Maximum time to finish job")
    parser.add_argument('--n-gpus', type=str, required=True, help="Number of GPUs")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--id-name', type=str, required=True, help="Experiment ID name")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--results-dir', type=str, required=True, help="Path to the results directory")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the output directory")

    args = parser.parse_args()

    submit_slurm_jobs(
        template=template_job,
        config=args.config,
        id_name=args.id_name,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        start_lr=0.0001,
        epochs=800,
        logging=50,
        factor_lr=2,
        steps_lr=1,
        n_gpus=args.n_gpus,
        hours=int(args.hours),
        n_cpus=10*int(args.n_gpus)
    )