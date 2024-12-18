# kgw, gen, gsm8k
export CUDA_VISIBLE_DEVICES=4;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method kgw \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --max_new_toks 200 \
    --save_generations \
    --outputs_dir ./outputs/gsm8k_kgw_deepseek_outputs \
    --save_generations_path machine_out.json \
    --generation_only \
    --gamma 0.5 \
    --delta 2 > logs/kgw-gsm8k-gen-deepseek-01.log 2>&1 &

# sweet, gen, gsm8k
export CUDA_VISIBLE_DEVICES=3;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method sweet \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --max_new_toks 200 \
    --save_generations \
    --outputs_dir ./outputs/gsm8k_sweet_deepseek_outputs \
    --save_generations_path machine_out.json \
    --generation_only \
    --gamma 0.5 \
    --ent_thresh 0.6 \
    --delta 2 > logs/sweet-gsm8k-gen-deepseek-01.log 2>&1 &


# kgw, detect, gsm8k
export CUDA_VISIBLE_DEVICES=0;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method kgw \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --metric_output_path evaluation_results.json \
    --load_generations_path ./outputs/gsm8k_kgw_deepseek_outputs/machine_out.json \
    --outputs_dir ./outputs/gsm8k_kgw_deepseek_outputs \
    --by_topk 2000 \
    --gamma 0.5 \
    --delta 2 > logs/kgw-gsm8k-eval-deepseek-01.log 2>&1 &


# sweet, detect, gsm8k
export CUDA_VISIBLE_DEVICES=5;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method sweet \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --metric_output_path evaluation_results.json \
    --load_generations_path ./outputs/gsm8k_sweet_deepseek_outputs/machine_out.json \
    --outputs_dir ./outputs/gsm8k_sweet_deepseek_outputs \
    --by_topk 2000 \
    --ent_thresh 0.6 \
    --gamma 0.5 \
    --delta 2 > logs/sweet-gsm8k-eval-deepseek-01.log 2>&1 &

# kgw, detect, human, gsm8k
export CUDA_VISIBLE_DEVICES=4;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method kgw \
    --detect_human_code \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --metric_output_path human_results.json \
    --load_generations_path ./outputs/gsm8k_kgw_deepseek_outputs/machine_out.json \
    --outputs_dir ./outputs/gsm8k_kgw_deepseek_outputs \
    --by_topk 2000 \
    --gamma 0.5 \
    --delta 2 > logs/human-kgw-gsm8k-eval-deepseek-01.log 2>&1 &


# sweet, detect, human, gsm8k
export CUDA_VISIBLE_DEVICES=3;
nohup python main.py \
    --model /data/deepseek-math-7b-instruct \
    --use_auth_token \
    --task gsm8k \
    --gen_method sweet \
    --detect_human_code \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 512 \
    --metric_output_path human_results.json \
    --load_generations_path ./outputs/gsm8k_sweet_deepseek_outputs/machine_out.json \
    --outputs_dir ./outputs/gsm8k_sweet_deepseek_outputs \
    --by_topk 2000 \
    --ent_thresh 0.6 \
    --gamma 0.5 \
    --delta 2 > logs/human-sweet-gsm8k-eval-deepseek-01.log 2>&1 &

###################performance###################
# calculate detection performance
python calculate_auroc_tpr.py \
    --human_fname ./outputs/gsm8k_kgw_deepseek_outputs/human_results.json \
    --machine_fname ./outputs/gsm8k_kgw_deepseek_outputs/evaluation_results.json \
    --min_length 15 \
    --max_length 999999 \
    --gen_method kgw > res/gsm8k_kgw_deepseek_res

python calculate_auroc_tpr.py \
    --human_fname ./outputs/gsm8k_sweet_deepseek_outputs/human_results.json \
    --machine_fname ./outputs/gsm8k_sweet_deepseek_outputs/evaluation_results.json \
    --min_length 15 \
    --max_length 999999 \
    --gen_method sweet > res/gsm8k_sweet_deepseek_res