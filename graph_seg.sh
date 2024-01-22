for ((i=1;i<=4;i++))
do
  #### pre_process
  python graph_process.py --fold $i --Direct_parameter "Low_resolution_4_Dice"
  python graph_process.py --fold $i --Direct_parameter "Hig_resolution_4_Dice"

  #### pre_segmentation
  python graph_seg.py --fold $i --Direct_parameter "Low_resolution_4_Dice"
  python graph_seg.py --fold $i --Direct_parameter "High_resolution_4_Dice"
done