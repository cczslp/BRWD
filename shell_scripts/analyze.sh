export CUDA_VISIBLE_DEVICES=6;
nohup python analysis.py \
    --precision fp16 \
    --gamma 0.5 \
    --delta 2.0 \
    --by_topk 2000 > logs/analyze_by_01.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0;
nohup python analysis.py \
    --target comp \
    --precision fp16 \
    --gamma 0.25 \
    --delta 2.0 \
    --by_topk 2000 > logs/analyze_by_02.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2;
nohup python analysis.py \
    --target score \
    --method by \
    --generations_dir outputs/mbpp_kgw_outputs \
    --precision fp16 \
    --gamma 0.5 \
    --delta 2.0 \
    --plt_lb -25 \
    --plt_ub 25 \
    --by_topk 2000 > logs/analyze_by_03.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2;
nohup python analysis.py \
    --target score \
    --method ewd \
    --generations_dir outputs/mbpp_kgw_outputs \
    --precision fp16 \
    --gamma 0.5 \
    --delta 2.0 \
    --plt_lb -7.535 \
    --plt_ub 11 \
    --by_topk 2000 > logs/analyze_ewd_03.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python analysis.py \
    --target score \
    --method ewd \
    --logp_cache_path pkl/mbpp_gr_logp.pkl \
    --generations_dir outputs/mbpp_kgw_outputs \
    --precision fp16 \
    --gamma 0.5 \
    --delta 2.0 \
    --by_topk 2000 > logs/analyze_by_04.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4;
nohup python prompt_effect_analysis.py \
    --task mbpp \
    --limit 50 \
    --factor confidence \
    --out_dir mbpp_kgw_outputs \
    --model /data/starcoder2-7b \
    --precision fp16 \
    --gamma 0.5 \
    --delta 2.0 \
    --by_topk 2000 > logs/penalty_analysis_mbpp_01.log 2>&1 &

# wii
export CUDA_VISIBLE_DEVICES=0;
nohup python analysis.py \
    --task_name mbpp \
    --target wii \
    --precision fp16 \
    --generations_dir outputs/mbpp_kgw_outputs \
    --gamma 0.5 \
    --delta 2.0 > logs/wii_analysis_mbpp_01.log 2>&1 &

