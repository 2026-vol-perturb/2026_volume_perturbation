#!/bin/bash

# ===== CONFIGURABLE VARIABLES ===== #

AVERAGE_CHECKPOINTS=15

# Optional checkpoint located in ./train_outs:
INIT_CHECKPOINT=
# For fine-tuning:
# INIT_CHECKPOINT=epoch_44.pt

NCPUS=2 # at least 2

# 500 h:
NGPUS=1
# 11800 h:
# NGPUS=2

PREFETCH_FACTOR=20

export WENET_DIR=$PWD/wenet_train

# ===== END CONFIGURABLE VARIABLES ===== #

log() {
    printf "%(%F %H:%M:%S)T    %s\n" -1 "$*" >&2
}

# Accepts an optional message to display.
# shellcheck disable=SC2120
fail() {
    # This needs to be the first line called after the statement that resulted in an error.
    local msg="Failed at (${BASH_SOURCE[1]}:${BASH_LINENO[0]})${FUNCNAME[1]:+" in ${FUNCNAME[1]}()"} (last exit status: ${PIPESTATUS[*]}) (workdir: $PWD)${*+": $*"}"
    
    log "$msg"

    # It is the current shell that exits, so the FAILED variable will be available in its exit trap handler.
    # shellcheck disable=SC2034
    FAILED="failed"
    exit 1
}

export BUILD_DIR=$WENET_DIR/runtime/libtorch/build
export OPENFST_BIN=$BUILD_DIR/../fc_base/openfst-build/src
export PATH=$BUILD_DIR/bin:$BUILD_DIR/kaldi:$OPENFST_BIN/bin:$PATH

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH

log "Generating raw file lists"

for set in "train" "cv"; do
    python "$WENET_DIR/tools/make_raw_list.py" \
        "$set/wav.scp" \
        "$set/text" \
        "$set/data.list" || fail "$set"
done

log "Training"

if [[ $INIT_CHECKPOINT ]]; then
    log "Starting from checkpoint $INIT_CHECKPOINT"
fi

mkdir -p train_outs || fail

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NGPUS" \
    --standalone \
    "$WENET_DIR/wenet/bin/train.py" \
    --config config.yaml \
    --data_type raw \
    --train_data "train/data.list" \
    --cv_data "cv/data.list" \
    ${INIT_CHECKPOINT:+--checkpoint "train_outs/$INIT_CHECKPOINT"} \
    --model_dir "train_outs" \
    --ddp.dist_backend "nccl" \
    --num_workers "$((NCPUS - 1))" \
    --pin_memory \
    --train_engine "torch_ddp" \
    --use_amp \
    --prefetch "$PREFETCH_FACTOR" || fail

log "Computing average checkpoint"

# Remove the symlink.
# If this is not performed, the final checkpoint is included twice in the average.
rm "train_outs/final.pt" || fail

exported_checkpoint=avg_$AVERAGE_CHECKPOINTS.pt
# The script averages the last N epochs picked based on their filesystem timestamp.
python "$WENET_DIR/wenet/bin/average_model.py" \
    --dst_model "train_outs/$exported_checkpoint" \
    --src_path "train_outs"  \
    --num "$AVERAGE_CHECKPOINTS" || fail

log "Successfully finished"

exit 0