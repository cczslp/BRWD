# generation, kgw, humaneval
export CUDA_VISIBLE_DEVICES=4;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task humaneval \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --save_generations \
    --outputs_dir ./outputs/humaneval_kgw_outputs \
    --save_generations_path machine_code.json \
    --generation_only \
    --gamma 0.5 \
    --delta 2 > logs/kgw-humaneval-gen-01.log 2>&1 &

# detect watermarked output, humaneval, kgw
export CUDA_VISIBLE_DEVICES=4;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task humaneval \
    --gen_method kgw \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --load_generations_path ./outputs/humaneval_kgw_outputs/machine_code.json \
    --outputs_dir ./outputs/humaneval_kgw_outputs \
    --by_topk 2000 \
    --gamma 0.5 \
    --delta 2 > logs/kgw-humaneval-eval-01.log 2>&1 &

# detect human output, humaneval, kgw
export CUDA_VISIBLE_DEVICES=7;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --gen_method kgw \
    --task humaneval \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --detect_human_code \
    --outputs_dir ./outputs/humaneval_kgw_outputs \
    --metric_output_path human_results.json \
    --gamma 0.5 \
    --delta 2 \
    --by_topk 2000 > logs/human-humaneval-eval-kgw-01.log 2>&1 &


# generation, sweet, humaneval
export CUDA_VISIBLE_DEVICES=7;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task humaneval \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --gen_method sweet \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --save_generations \
    --outputs_dir ./outputs/humaneval_sweet_outputs \
    --save_generations_path machine_code.json \
    --generation_only \
    --ent_thresh 0.6 \
    --gamma 0.5 \
    --delta 2 > logs/sweet-humaneval-gen-01.log 2>&1 &

# detect watermarked output, humaneval, sweet
export CUDA_VISIBLE_DEVICES=4;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task humaneval \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --metric_output_path evaluation_results.json \
    --load_generations_path ./outputs/humaneval_sweet_outputs/machine_code.json \
    --outputs_dir ./outputs/humaneval_sweet_outputs \
    --gen_method sweet \
    --by_topk 2000 \
    --gamma 0.5 \
    --ent_thresh 0.6 \
    --delta 2 > logs/sweet-humaneval-eval-01.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task humaneval \
    --gen_method sweet \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --detect_human_code \
    --outputs_dir ./outputs/humaneval_sweet_outputs \
    --metric_output_path human_results.json \
    --gamma 0.5 \
    --delta 2 \
    --by_topk 2000 \
    --ent_thresh 0.6 > logs/human-humaneval-eval-sweet-01.log 2>&1 &

# analyze data
python calculate_auroc_tpr.py \
    --human_fname ./outputs/humaneval_kgw_outputs/human_results.json \
    --machine_fname ./outputs/humaneval_kgw_outputs/evaluation_results.json \
    --min_length 15 \
    --max_length 999999 \
    --gen_method kgw > res/humaneval_kgw_res