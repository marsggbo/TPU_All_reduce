export TPU_NAME=tpu32
export BS=4096
export SCALE=$1
export DATA_DIR=gs://hkbuautoml/data
export MODEL_DIR=gs://hkbuautoml/mnist_tpu32_bs${BS}_scale${SCALE}
export TPU_ZONE=us-central1-a
export NUM_SHARDS=32


python mnist_tpu.py \
	--tpu=$TPU_NAME \
	--tpu_zone=$TPU_ZONE \
	--scale=$SCALE \
	--num_shards=$NUM_SHARDS \
	--batch_size=$BS \
	--data_dir=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--use_tpu=True \
	--iterations=500 \
	--train_steps=5000 \
	--use_bfloat16=False \
	--enablr_predict=False \
	--eval_steps=0 \