DATA_DIR1="recipeQA"
DATA_NAME1="recipeqa"
TASK_TYPE="pretrain"
MODEL_TYPE="clip"
VISION_MODEL="resnet50"

OUTPUT_ROOT="your/output/root"
OUTPUT_NAME="your/output/name"

python3 -m trainers.run_pretraining \
  --model_name_or_path "pretrained_models/roberta/large" \
  --config_name "bert-base-uncased" \
  --config_name "roberta-large" \
  --tokenizer_name "bert-base-uncased" \
  --tokenizer_name "roberta-large" \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 20.0 \
  --max_seq_length 300 \
  --per_seq_max_length 60 \
  --data_dirs "data/${DATA_DIR1}" \
  --data_names ${DATA_NAME1} \
  --max_story_length 5 \
  --output_root "${OUTPUT_ROOT}" \
  --output_dir "${OUTPUT_NAME}" \
  --task_type ${TASK_TYPE} \
  --order_criteria "loose" \
  --overwrite_output_dir \
  --multimodal \
  --multimodal_model_type ${MODEL_TYPE} \
  --vision_model "${VISION_MODEL}" \
  --save_steps 2000 \
  --logging_steps 500 \
  --max_eval_steps 200 \
  --iters_to_eval 20000 \
  --warmup_steps 500 \
  --eval_splits "test-human_annot_only" \
  --train_split "train-human_annot" \
  --mlm_probability 0.1 \
  --multimodal_pretrain_objectives "image_swapping" "patch_based_image_swapping" "patch_based_mrm_classification" \
  # --clip_visual_model_weights "your/clip/model/path/pytorch_model.bin" \
