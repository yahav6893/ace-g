model_name="ace_g_5min"
dataset="indoor6"
scene="scene2a"

# Download model
if [ ! -f "ace_g_pretrained.pt" ]; then
    echo "Downloading model..."
    wget https://storage.googleapis.com/niantic-lon-static/research/ace-g/ace_g_pretrained.pt
else
    echo "Model already downloaded. Skipping."
fi

# Download indoor6 dataset
if [ ! -d "datasets/indoor6" ]; then
    echo "Downloading indoor6 dataset..."
    cd datasets && python setup_indoor6.py && cd ..
else
    echo "Indoor6 dataset already downloaded. Skipping."
fi

# Visualize mapping on mapping sequence
echo "Running mapping..."
python -m ace_g.train_single_scene \
    --config "${model_name}.yaml" \
    --dataset.rgb_files "datasets/${dataset}/${scene}/train/rgb/*.jpg" \
    --dataset.pose_files "datasets/${dataset}/${scene}/train/poses/*.txt" \
    --session_id "${model_name}-${dataset}-${scene}" \
    --rerun_spawn True \
    --use_rerun True

# Visualize localization on query sequence
echo "Running localization..."
python -m ace_g.vis_localization \
    --config "./outputs/${model_name}-${dataset}-${scene}_map.yaml" \
    --dataset.rgb_files "datasets/${dataset}/${scene}/test/rgb/*.jpg" \
    --dataset.pose_files "datasets/${dataset}/${scene}/test/poses/*.txt" \
    --rerun_spawn True \
    --use_rerun True
