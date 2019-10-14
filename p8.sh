export SCALE=$1
export BS=1024
export TPU_NAME=tpuv2n8
export TPU_ZONE=us-central1-f
export NS=8
export DIR=gs://hkbuautoml/mnist_${TPU_NAME}_bs${BS}_scale${SCALE}

echo $DIR

capture_tpu_profile \
    --tpu=$TPU_NAME \
    --tpu_zone=$TPU_ZONE \
    --logdir=${DIR} \
