export TASK_NAME=mnli

python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/