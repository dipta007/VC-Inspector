# You can refer to `https://github.com/QwenLM/Qwen2.5-VL` for the meaning of the `VIDEO_MAX_PIXELS` parameter.

source .venv/bin/activate

MODEL_ID=$1
HF_DATASET="dipta007/ActivityNet-FG-It"
HF_SUBSET="balanced"
LOCAL_DATA_DIR="data/ActivityNet-FG-It"

# Download dataset locally if not already present
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    mkdir -p "$LOCAL_DATA_DIR"
    python -c "
from datasets import load_dataset
ds = load_dataset('$HF_DATASET', '$HF_SUBSET')
for split in ds:
    ds[split].to_json(f'$LOCAL_DATA_DIR/{split}.jsonl')
print('Dataset downloaded to $LOCAL_DATA_DIR')
"
fi

# Download and extract frames if not already present
if [ ! -d "frames" ]; then
    python -c "
from huggingface_hub import hf_hub_download
print('Downloading frames.zip...')
path = hf_hub_download(repo_id='$HF_DATASET', filename='frames.zip', repo_type='dataset')
print(f'Downloaded to {path}')
import zipfile
print('Extracting frames...')
with zipfile.ZipFile(path, 'r') as z:
    z.extractall('.')
print('Frames extracted.')
"
fi

TRAIN_DATASET="$LOCAL_DATA_DIR/train.jsonl"
VAL_DATASET="$LOCAL_DATA_DIR/val.jsonl"
MODEL_NAME=$(echo $MODEL_ID | cut -d '/' -f 2)
MODEL_TYPE=qwen2_5_vl

OUTPUT_DIR="outputs/$MODEL_NAME-lora"

nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())")

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * $nproc_per_node)))

echo Training $MODEL_NAME on $CUDA_VISIBLE_DEVICES
echo Saving to $OUTPUT_DIR
echo nproc_per_node: $nproc_per_node
echo Global batch size: $GLOBAL_BATCH_SIZE
echo Batch per device: $BATCH_PER_DEVICE
echo Grad accumulation steps: $GRAD_ACCUM_STEPS
echo Train dataset: $TRAIN_DATASET
echo Val dataset: $VAL_DATASET
echo Model name: $MODEL_NAME
echo Model id: $MODEL_ID
echo Model type: $MODEL_TYPE
echo Output dir: $OUTPUT_DIR
MASTER_PORT=$(expr 10000 + $RANDOM % 20000) \
NPROC_PER_NODE=$nproc_per_node \
MAX_NUM=4 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=64 \
swift sft \
    --model $MODEL_ID \
    --model_type $MODEL_TYPE \
    --use_hf \
    --dataset $TRAIN_DATASET \
    --val_dataset $VAL_DATASET \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_ratio 0.05 \
    --lr_scheduler_kwargs '{"min_lr": 1e-5}' \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --save_strategy epoch \
    --eval_strategy no \
    --eval_steps 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --report_to none \
    --run_name "$MODEL_NAME-lora" \
    --load_from_cache_file false \
    --logging_first_step true \
    --gradient_checkpointing true \
    --add_version false
