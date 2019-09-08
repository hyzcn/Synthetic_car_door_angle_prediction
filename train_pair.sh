SAVE_FOLDER="test"
python cnn_pair.py --command train --model-name resnet --part-name all \
                --batch-size 64 --num-epoch 2 --add-crop False --num-images 97200 \
                --sample-iter 6 --feature-extract False --data-range 60 \
                --train-dir datasets/train/preset_car_data/ \
                --crop-dir datasets/train/preset_car_crop/ \
                --test-dir datasets/all_test/preset_all_random/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \