for ((i=1;i<=4;i++))
do
  python tree_process.py --fold $i --patch_size 16 --z_size 4  --Direct_parameter "Low_resolution_4_Dice"
  python tree_process.py --fold $i --patch_size 16 --z_size 4  --Direct_parameter "High_resolution_4_Dice"
  python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvLSTM" --Direct_parameter "Low_resolution_4_Dice"
  python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvGRU" --Direct_parameter "Low_resolution_4_Dice"
  python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 8 --model "TreeConvGRU" --Direct_parameter "Low_resolution_4_Dice"
  python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvGRU" --Direct_parameter "High_resolution_4_Dice"

done