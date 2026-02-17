# --selective true \
python inference.py \
    --datapath dataset/all_questions_descfilter_bg.json \
    --return_logits true \
    --selective true \
    --bare_question true \
    --lora_ckpt ./qwen25_omni_sft_out/checkpoint-18000
