# detect watermarked output, mbpp, kgw
export CUDA_VISIBLE_DEVICES=1;
nohup python main.py \
    --model /data/starcoder2-7b \
    --use_auth_token \
    --task mbpp \
    --gen_method kgw \
    --temperature 0.2 \
    --precision fp16 \
    --batch_size 1 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --load_generations_path ./outputs/mbpp_kgw_outputs/machine_code.json \
    --outputs_dir ./outputs/mbpp_kgw_outputs \
    --metric_output_path sub_eval_results.json \
    --attacked_gen_path var_replaced_code.json \
    --by_topk 2000 \
    --gamma 0.5 \
    --delta 2 > logs/kgw-mbpp-eval-sub-01.log 2>&1 &


python calculate_auroc_tpr.py \
    --human_fname ./outputs/mbpp_kgw_outputs/human_results.json \
    --machine_fname ./outputs/mbpp_kgw_outputs/sub_eval_results.json \
    --min_length 15 \
    --max_length 999999 \
    --gen_method kgw > res/mbpp_kgw_res_sub_atk