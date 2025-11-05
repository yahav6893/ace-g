run_single_scene(model_name, dataset, scene)
  python -m ace_g.train_single_scene \
  --config "${model_name}.yaml" \
  --dataset.rgb_files "datasets/${dataset}/${scene}/train/rgb/*.jpg" \
  --dataset.pose_files "datasets/${dataset}/${scene}/train/poses/*.txt" \
  --session_id "${model_name}-${dataset}-${scene}"

  python -m ace_g.register_images \
    --config "./outputs/${model_name}-${dataset}-${scene}_map.yaml" \
    --dataset.rgb_files "datasets/${dataset}/${scene}/test/rgb/*.jpg"
    
  python -m ace_g.eval_poses \
    --config "./outputs/${model_name}-${dataset}-${scene}_reg.yaml" \
    --gt_pose_files "datasets/${dataset}/${scene}/test/poses/*.txt"
end

run_rio10(model_name) {
  dataset="rio10"
  scenes=("scene01" "scene02" "scene03" "scene04" "scene05" "scene06" "scene07" "scene08" "scene09" "scene10")
  for scene in "${scenes[@]}"; do
    run_single_scene(model_name, dataset, scene)
  done
}

run_indoor6(model_name) {
  dataset="indoor6"
  scenes=("scene1" "scene2a" "scene3" "scene4a" "scene5" "scene6")
  for scene in "${scenes[@]}"; do
    run_single_scene(model_name, dataset, scene)
  done
}

model_name="ace_g_5min"
run_single_scene(model_name, "rio10", "scene06")
# run_rio10(model_name)
# run_indoor6(model_name)