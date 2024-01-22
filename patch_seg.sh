for ((i=1;i<=4;i++))
do

  python patch_process.py --fold $i --patch_size 32 --pools 32 --Direct_parameter "Low_resolution_4_Dice"


  python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 ##experment 1
  python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 --flip_prob 0.5 --rotate_prob 0.5 ##experment 2
  python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 --flip_prob 0 --rotate_prob 0 ##experment 3
  python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 1 --load_num 0 --batch_size 64 ##experment 4 add frangi

  python patch_process.py --fold $i --patch_size 64 --pools 32

  python patch_seg.py --fold $i --patch_size 64 --pools 32 --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 10  ##experment5

done


#python patch_seg.py --fold 1 --patch_size 32 --pools 32  --num_workers 8 --is_train 0 --frangi 0 --load_num 0 --batch_size 64 ##experment 1