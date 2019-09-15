SAVE_FOLDER="loc_pre_texture"

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


CUDA_VISIBLE_DEVICES=2,3 python cnn_loc.py --command train --model-name resnet --part-name all \
                --batch-size 16 --num-epoch 50 --add-crop False --add-pre True \
                --sample-iter 600 --feature-extract False --data-range 60 --num-images 97200 \
                --train-dir datasets/train/preset_car_data_texture/ \
                --train-gt-dir gt_dict/preset_car_{}_gt.npy \
                --crop-dir datasets/train/preset_car_crop/ \
                --crop-gt-dir gt_dict/preset_car_crop_{}_gt.npy \
                --test-dir datasets/all_test/preset_test_random/ \
                --test-gt-dir gt_dict/preset_test_random_all_gt.npy \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_0.3_0.6_64.pkl \
                --plot-dir plots/$SAVE_FOLDER/ \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --train-name-dir ImageSets/preset_all_train_texture_sample.txt \
                --test-name-dir ImageSets/preset_all_test_texture_sample.txt > params/$SAVE_FOLDER/log