DATA_DIR1="wikihow"
DATA_NAME1="wikihow"
TASK_TYPE="pretrain"
MODEL_TYPE="clip"
VISION_MODEL="resnet50"

OUTPUT_ROOT="your/output/root"
OUTPUT_NAME="your/output/name"

python3 -m trainers.run_pretraining \
  --model_name_or_path "pretrained_models/bert/base_uncased" \
  --config_name "bert-base-uncased" \
  --tokenizer_name "bert-base-uncased" \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 50 \
  --per_seq_max_length 10 \
  --data_dirs "data/${DATA_DIR1}" \
  --data_names ${DATA_NAME1} \
  --max_story_length 5 \
  --output_root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_NAME}" \
  --task_type ${TASK_TYPE} \
  --order_criteria "loose" \
  --overwrite_output_dir \
  --multimodal \
  --multimodal_img_part \
  --multimodal_model_type ${MODEL_TYPE} \
  --vision_model "${VISION_MODEL}" \
  --save_steps 2000 \
  --logging_steps 500 \
  --max_eval_steps 200 \
  --iters_to_eval 20000 \
  --warmup_steps 1000 \
  --eval_splits "test-acl22_human" \
  --train_split "train-acl22" \
  --multimodal_pretrain_objectives "patch_based_mrm_classification" \
