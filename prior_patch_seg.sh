for ((i=1;i<=1;i++))
do
  # dilation prior segmentation
  python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 --loss "Dice_dilation"
  python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice_dilation" --pools 32
  python prior_patch_process.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice_dilation"

  python prior_patch_seg.py --fold $i --patch_size 16 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 512 --Direct_parameter "Low_resolution_4_Dice_dilation"
  python prior_patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice_dilation"
  python prior_patch_seg.py --fold $i --patch_size 64 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 8 --Direct_parameter "Low_resolution_4_Dice_dilation"
  python prior_seg_ensemble.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice_dilation" --global_seg "Low_resolution_4_Dice"
  python prior_seg_ensemble.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice_dilation" --global_seg "High_resolution_4_Dice"

  # normal prior segmentation
  python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 --loss "Dice" # 如果direct 已经训练可以不重新训练
  python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice" --pools 32
  python prior_patch_process.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice"

  python prior_patch_seg.py --fold $i --patch_size 16 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 512 --Direct_parameter "Low_resolution_4_Dice"
  python prior_patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice"
  python prior_patch_seg.py --fold $i --patch_size 64 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 8 --Direct_parameter "Low_resolution_4_Dice"

  python prior_seg_ensemble.py --fold $i --Direct_parameter "Low_resolution_4_Dice" --global_seg "Low_resolution_4_Dice"
  python prior_seg_ensemble.py --fold $i --Direct_parameter "Low_resolution_4_Dice" --global_seg "High_resolution_4_Dice"
done
