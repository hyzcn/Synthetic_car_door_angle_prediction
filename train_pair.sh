SAVE_FOLDER="crop"

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

CUDA_VISIBLE_DEVICES=0,1 python cnn_pair.py --command train --model-name resnet --part-name all \
                --batch-size 16 --num-epoch 50 --add-crop True --num-images 97200 \
                --sample-iter 600 --feature-extract False --data-range 60 \
                --train-dir datasets/train/preset_car_data/ \
                --crop-dir datasets/train/preset_car_crop/ \
                --test-dir datasets/all_test/preset_test_random/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --train-name-dir ImageSets/preset_all_train_sample.txt \
                --test-name-dir ImageSets/preset_all_test_sample.txt > params/$SAVE_FOLDER/log