SAVE_FOLDER="norm_final"

if [ ! -d params/$SAVE_FOLDER  ];then
  mkdir params/$SAVE_FOLDER
else
  echo dir exist
fi
if [ ! -d plots/$SAVE_FOLDER  ];then
  mkdir plots/$SAVE_FOLDER
else
  echo dir exist
fi
if [ ! -d outputs/$SAVE_FOLDER  ];then
  mkdir outputs/$SAVE_FOLDER
else
  echo dir exist
fi
if [ ! -d htmls/$SAVE_FOLDER  ];then
  mkdir htmls/$SAVE_FOLDER
else
  echo dir exist
fi

CUDA_VISIBLE_DEVICES=0,1 python cnn_norm.py --command train --model-name resnet --part-name all \
                --batch-size 16 --num-epoch 50 --num-images 97200 \
                --train-num 45000 --test-num 9720 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data_final/ \
                --test-dir datasets/all_test/preset_all_same/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_floor_norm_new.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}_floor_norm_new.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}_floor_norm_new.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}_floor_norm_new.txt \
                --train-gt-dir gt_dict/preset_car_all_gt.npy \
                --test-gt-dir gt_dict/preset_car_all_gt.npy \
                --train-name-dir ImageSets/preset_all_train_final_norm.txt \
                --test-name-dir ImageSets/preset_all_test_final_norm.txt \
                --test-spatial False > params/$SAVE_FOLDER/log_floor_new

# CUDA_VISIBLE_DEVICES=0,1 python cnn_norm_seg.py --command train --model-name resnet --part-name all \
#                 --batch-size 16 --num-epoch 50 --num-images 97200 \
#                 --train-num 45000 --test-num 9720 --lam 0.5 \
#                 --feature-extract False --data-range 60 --seg-classes 9 \
#                 --data-dir datasets/train/preset_car_data/ \
#                 --train-gt-dir gt_dict/preset_car_all_gt.npy \
#                 --train-seg-gt-dir segsets/train/preset_car_data/ \
#                 --test-dir datasets/all_test/shapenet_test_all/ \
#                 --test-gt-dir gt_dict/shapenet_test_all_all_gt.npy \
#                 --test-seg-gt-dir segsets/all_test/preset_all_same/ \
#                 --model-dir params/{}_ft_{}_floor_norm_new_seg_0.5.pkl \
#                 --plot-dir plots/{}_ft_{}_floor_norm_new_seg_0.5.jpg \
#                 --output-dir outputs/{}_ft_{}_floor_norm_new_seg_0.5.txt \
#                 --html-dir htmls/{}_ft_{}_floor_norm_new_seg_0.5.txt \
#                 --train-name-dir ImageSets/preset_all_train_final_norm.txt \
#                 --test-name-dir ImageSets/preset_all_test_final_norm.txt \
#                 --test-spatial False > params/log_floor_new_seg_0.5
    
# test-spatial means using test set
# else, read from file

