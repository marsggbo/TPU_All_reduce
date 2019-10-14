export SCALE=$1
export BS=65536
export TPU_NAME=tpu512
export TPU_ZONE=europe-west4-a
export DIR=gs://hkbuautoml/mnist_${TPU_NAME}_bs${BS}_scale${SCALE}

echo $DIR

capture_tpu_profile \
    --tpu=$TPU_NAME \
    --tpu_zone=$TPU_ZONE \
    --logdir=${DIR} \
