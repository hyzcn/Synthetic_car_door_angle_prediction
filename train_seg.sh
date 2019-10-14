SAVE_FOLDER="loc_pre_seg_texture_0.5_goon"

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


CUDA_VISIBLE_DEVICES=0,1 python cnn_seg.py --command train --model-name resnet --part-name all \
                --batch-size 16 --num-epoch 50 --add-crop False --add-pre True --seg-classes 9 \
                --sample-iter 600 --feature-extract False --data-range 60 --num-images 97200 \
                --lam 0.5 \
                --train-dir datasets/train/preset_car_data_texture/ \
                --train-gt-dir gt_dict/preset_car_{}_gt.npy \
                --train-seg-gt-dir segsets/train/preset_car_data/ \
                --crop-dir datasets/train/preset_car_crop_texture/ \
                --crop-gt-dir gt_dict/preset_car_crop_texture_{}_gt.npy \
                --test-dir datasets/all_test/preset_all_same/ \
                --test-gt-dir gt_dict/preset_all_same_all_gt.npy \
                --test-seg-gt-dir segsets/all_test/preset_all_same/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_0.3_0.6_64.pkl \
                --plot-dir plots/$SAVE_FOLDER/ \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --test-texture True \
                --train-name-dir ImageSets/preset_all_train_texture_norm.txt \
                --test-name-dir ImageSets/preset_all_test_texture_norm.txt > params/$SAVE_FOLDER/log

# if test-baseline and test-texture, the data dir is the same as train dir
# test-texture meams read names from file
# else, the data dir is test dir