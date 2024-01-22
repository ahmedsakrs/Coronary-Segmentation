for ((i=1;i<=4;i++))
do
  python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 1 --batch_size 1
  python direct_seg.py --fold $i --channel 4 --model "FCN_AG" --rl 1 --batch_size 1

  python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 2 --batch_size 2
  python direct_seg.py --fold $i --channel 12 --model "FCN" --rl 2 --batch_size 2

  python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8
  python direct_seg.py --fold $i --channel 12 --model "FCN" --rl 3 --batch_size 8

done

