SAVE_FOLDER="sigmoid_texture"

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

CUDA_VISIBLE_DEVICES=1 python cnn_pair.py --command train --model-name resnet --part-name all \
                --batch-size 8 --num-epoch 100 --add-crop False --num-images 97200 \
                --sample-iter 600 --feature-extract False --data-range 60 \
                --train-dir datasets/train/preset_car_data_texture/ \
                --crop-dir datasets/train/preset_car_crop/ \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \