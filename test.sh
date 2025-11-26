export CUDA_VISIBLE_DEVICES=0

pip install soundfile
pip install numpy==1.26.4
pip install qwen-omni-utils

# --selective true \
python inference.py \
    --datapath dataset/all_questions_descfilter_bg.json \
    --return_logits true \
    --selective true \
    --bare_question true \
    --lora_ckpt ./qwen25_omni_sft_out/checkpoint-18000
