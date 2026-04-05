#!/bin/bash

algo=(sl es wq sdq)
comp=(csews29 csews28 csews25 csews27)
impr=(pa do tl)

infer_algo=(sarsa_lambda exp_sarsa watkins_qlambda stoch_dyna_q)
infer_impv=(prev_action decay_obs target_lock)

SUB_DIR="$HOME/obelix/submissions"
CFG_DIR="$SUB_DIR/configs"
SRC_DIR="$HOME/obelix/src/agents/phase_2"

mkdir -p "$CFG_DIR"

n=4

# Create dirs + copy agent + fetch config

for i in {1..4}; do
  for j in {1..3}; do
    for k in {1..4}; do

      name="${algo[i-1]}${impr[j-1]}_${k}"
      dir="$SUB_DIR/${n}.${name}"

      # Make directory 
      mkdir -p "$dir"
      echo "made directory $dir"

      # Copy agent
      src_py="$SRC_DIR/${infer_algo[i-1]}_${infer_impv[j-1]}_infer.py"
      dst_py="$dir/agent.py"

      cp "$src_py" "$dst_py"
      echo "copied agent -> $dst_py"

      # Copy config (scp) 
      remote="hriddhitd25@image.cse.iitk.ac.in:~/obelix/models/${comp[i-1]}_${algo[i-1]}${impr[j-1]}_p2_config_${k}.json"
      local_cfg="$CFG_DIR/${n}.${name}.json"

      scp "$remote" "$local_cfg"
      echo "fetched config -> $local_cfg"

      # Copy weigths (scp) 
      weight_remote="hriddhitd25@image.cse.iitk.ac.in:~/obelix/models/${comp[i-1]}_${algo[i-1]}${impr[j-1]}_p2_final_${k}_weights.pth"
      weight_local="$dir/weights.pth"

      scp "$weight_remote" "$weight_local"
      echo "fetched weights -> $weight_local"

      # Increment counter
      n=$((n + 1))

    done
  done
done
