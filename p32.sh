export SCALE=$1
export BS=4096
export TPU_NAME=tpu32
export TPU_ZONE=us-central1-a
export NS=32
export DIR=gs://hkbuautoml/mnist_${TPU_NAME}_bs${BS}_scale${SCALE}

echo $DIR

capture_tpu_profile \
    --tpu=$TPU_NAME \
    --tpu_zone=$TPU_ZONE \
    --logdir=${DIR} \
