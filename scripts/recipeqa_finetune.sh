DATA_NAME="recipeQA"
TASK_NAME="recipeqa"
MODEL_TYPE="clip"
VISION_MODEL="resnet50"

OUTPUT_ROOT="your/output/root"
OUTPUT_NAME="your/output/name"

python3 -m trainers.train \
  --model_name_or_path "your/pretrained/model/path/checkpoint-[ITER]" \
  --config_name "bert-base-uncased" \
  --config_name "roberta-large" \
  --tokenizer_name "bert-base-uncased" \
  --tokenizer_name "roberta-large" \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 4.0 \
  --max_seq_length 300 \
  --per_seq_max_length 60 \
  --data_dir "data/${DATA_NAME}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_NAME}" \
  --task_name "${TASK_NAME}_hl_v1" \
  --order_criteria "loose" \
  --overwrite_output_dir \
  --multimodal \
  --multimodal_model_type ${MODEL_TYPE} \
  --train_split "train-human_annot" \
  --vision_model "${VISION_MODEL}" \
  --wrapper_model_type "berson" \
  --save_steps 2000 \
  --logging_steps 250 \
  --max_eval_steps 1000 \
  --iters_to_eval 16000 \
  --warmup_steps 100 \
  --train_split "train-acl22" \
  --eval_splits "test-acl_human" \
