export TPU_NAME=grpc://10.0.101.2:8470
export DATA_DIR=gs://hkbuautoml/data
export BS=1024
export SCALE=$1
export TPU_ZONE=us-central1-f
export MODEL_DIR=gs://hkbuautoml/mnist_tpuv2n8_bs${BS}_scale${SCALE}
echo $MODEL_DIR

python mnist_tpu.py \
	--tpu=$TPU_NAME \
	--scale=$SCALE \
	--tpu_zone=$TPU_ZONE \
	--batch_size=$BS \
	--data_dir=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--use_tpu=True \
	--iterations=5000 \
	--train_steps=50000 \
	--use_bfloat16=False \
	--enablr_predict=False \
	--eval_steps=0 \
