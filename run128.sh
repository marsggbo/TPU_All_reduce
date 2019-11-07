export TPU_NAME=tpu128
export BS=16384
export SCALE=$1
export DATA_DIR=gs://hkbuautoml/data
export MODEL_DIR=gs://hkbuautoml/mnist_${TPU_NAME}_bs${BS}_scale${SCALE}
export TPU_ZONE=us-central1-a
export NUM_SHARDS=128


python mnist_tpu.py \
	--tpu=$TPU_NAME \
	--tpu_zone=$TPU_ZONE \
	--scale=$SCALE \
	--num_shards=$NUM_SHARDS \
	--batch_size=$BS \
	--data_dir=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--use_tpu=True \
	--iterations=50000 \
    --train_steps=5000000 \
	--use_bfloat16=False \
	--enablr_predict=False \
	--eval_steps=0 \
