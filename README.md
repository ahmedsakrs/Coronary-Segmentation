
## Paper: Coronary Arteries Segmentation Based on 3D FCN With Attention Gate and Level Set Function

####baseline command: 
```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8  
```
####分辨率对比:  

```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 1 --batch_size 1  
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 2 --batch_size 2  
rl=1 or 2 or 3  
1:input size [512,512,256]   
2:input size [256,256,128]   
3:input size [128,128,64]   
```

####增加attention:  
```
python direct_seg.py --fold $i --channel 4 --model "FCN_AG" --rl 3 --batch_size 8
```

####增加通道:  

```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8  
python direct_seg.py --fold $i --channel 12 --model "FCN" --rl 3 --batch_size 2
```

## Paper:Coronary Artery Segmentation in Cardiac CT Angiography Using 3D Multi-Channel U-net
### Command
####预处理：  
```
python patch_process.py --fold $i --patch_size 32 --pools 32 --Direct_parameter "Low_resolution_4_Dice"  
python patch_process.py --fold $i --patch_size 64 --pools 32 --Direct_parameter "Low_resolution_4_Dice"  
保证在命令 python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 已经运行过，并对训练，测试进行推理,生成Low_resolution_4_Dice参数结果
```
####baseline:
```
python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice"  ##experment 1  
```
####数据增强影响： 
```
python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 --flip_prob 0.5 --rotate_prob 0.5  --Direct_parameter "Low_resolution_4_Dice"  
python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 64 --flip_prob 0 --rotate_prob 0 --Direct_parameter "Low_resolution_4_Dice"  
```
####增加frangi通道：
```
python patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --frangi 1 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice"  ##experment 4 add frangi
```
#### patch大小
```
python patch_seg.py --fold $i --patch_size 64 --pools 32 --num_workers 8 --is_train 1 --frangi 0 --load_num 0 --batch_size 10 --Direct_parameter "Low_resolution_4_Dice"   ##experment5
```
## Paper:Learning tree-structured representation for 3d coronary artery segmentation

```
保证在命令 python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 已经运行过，并对训练，测试进行推理,生成Low_resolution_4_Dice参数结果
python morphology_process.py --fold $i --Direct_parameter "Low_resolution_4_Dice"
```
####预处理

```
python tree_process.py --fold $i --patch_size 16 --z_size 4  --Direct_parameter "Low_resolution_4_Dice"
python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice_dilation" --pools 32  
```

####baseline
```
python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvGRU" --Direct_parameter "Low_resolution_4_Dice"
```
####model
```
python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvLSTM" --Direct_parameter "Low_resolution_4_Dice"
p
```

####patch
```
python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 8 --model "TreeConvGRU" --Direct_parameter "Low_resolution_4_Dice"
```
#### pre_segmentation
```
python tree_seg.py --gpu_index 0 --fold $i --patch_size 16 --z_size 4 --model "TreeConvGRU" --Direct_parameter "High_resolution_4_Dice"  
ps:先运行 python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 1 --batch_size 1 并对训练，测试集进行推理,生成High_resolution_4_Dice参数结果
```
## Paper:Graph convolutional networks for coronary artery segmentation in cardiac ct angiography
```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 并对训练，测试集进行推理，测试进行推理,生成Low_resolution_4_Dice参数结果  
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 1 --batch_size 1 并对训练，测试集进行推理,生成High_resolution_4_Dice参数结果
python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice_dilation" --pools 32  
python morphology_process.py --fold $i  --Direct_parameter "High_resolution_4_Dice_dilation" --pools 32  
```

#### pre_process
```
python graph_process.py --fold $i --Direct_parameter "Low_resolution_4_Dice"
```
#### pre_segmentation
```
python graph_seg.py --fold $i --Direct_parameter "Low_resolution_4_Dice"
python graph_seg.py --fold $i --Direct_parameter "High_resolution_4_Dice"
```

## Coarse to fine
#### normal prior
```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 --loss "Dice" # 如果direct 已经训练可以不重新训练  
python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice" --pools 32  
python prior_patch_process.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice"  

python prior_patch_seg.py --fold $i --patch_size 16 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 512 --Direct_parameter "Low_resolution_4_Dice"  
python prior_patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice"  
python prior_patch_seg.py --fold $i --patch_size 64 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 8 --Direct_parameter "Low_resolution_4_Dice"  
```
#### dilation prior
```
python direct_seg.py --fold $i --channel 4 --model "FCN" --rl 3 --batch_size 8 --loss "Dice_dilation"  
python morphology_process.py --fold $i  --Direct_parameter "Low_resolution_4_Dice_dilation" --pools 32  
python prior_patch_process.py --fold $i --pools 32 --Direct_parameter "Low_resolution_4_Dice_dilation"  

python prior_patch_seg.py --fold $i --patch_size 16 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 512 --Direct_parameter "Low_resolution_4_Dice_dilation"  
python prior_patch_seg.py --fold $i --patch_size 32 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 64 --Direct_parameter "Low_resolution_4_Dice_dilation"  
python prior_patch_seg.py --fold $i --patch_size 64 --pools 32  --num_workers 8 --is_train 1 --load_num 0 --batch_size 8 --Direct_parameter "Low_resolution_4_Dice_dilation"  
```
## Structure
```
project/
    -config/
        -config.yaml
    -data/
        -data_loader.py
        -.....
    -utils/
        -....
    -model/
        -net
        -.....
    -Intermediate_data/
        -Patch/
        -Tree/
        -Graph/
        -Prior_Patch/
    -result/
        -Direct/
        -Patch/
            -/Pre_seg_name
                -different_parameters_name/
                    -pre_label
                    -model_save
                    -result.csv 
        -Tree/
        -Graph/
        -Prior_Patch/
    -main.py
    -run.sh
```