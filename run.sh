pip install numpy==1.26.4
pip install qwen-omni-utils[decord] -U

torchrun --nproc_per_node=8 --master_port=12346 train.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --dataset dataset/synth_audio_questions_train_filter.json \
  --val_dataset dataset/all_questions_descfilter_bg.json \
  --output_dir ./qwen25_omni_sft_out \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --max_seq_length 4096 \
  --save_steps 2000 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --flash_attn true \
  --bf16 true \
