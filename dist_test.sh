CONFIG=$1

WORK_DIR=${WORK_DIR:-"./work_logs"}
GPUS=${GPUS:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# echo $GPUS 4
# echo $NNODES 1
# echo $NODE_RANK 0
# echo $PORT 29500
# echo $MASTER_ADDR 127.0.0.1


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/eval.py \
    --config $CONFIG \
    --arch reduced \
    --attn naclip \
    --std 5 \
    --pamr on \
    --work-dir $WORK_DIR \
    --launcher pytorch \
    ${@:4}