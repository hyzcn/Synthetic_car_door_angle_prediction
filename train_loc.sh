SAVE_FOLDER="test"
# only when SAVE_FOLDER is locaion, use False
ADD_PRE=True
python cnn_loc.py --command train --model-name resnet --part-name all \
                --batch-size 64 --num-epoch 2 --add-crop False --add-pre $ADD_PRE \
                --sample-iter 6 --feature-extract False --data-range 60 --num-images 97200 \
                --train-dir datasets/train/preset_car_data/ \
                --train-gt-dir gt_dict/preset_car_{}_gt.npy \
                --crop-dir datasets/train/preset_car_crop/ \
                --crop-gt-dir gt_dict/preset_car_crop_{}_gt.npy \
                --test-dir datasets/all_test/preset_all_random/ \
                --test-gt-dir gt_dict/preset_all_random_{}_gt.npy \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_0.3_0.6_64.pkl \
                --plot-dir plots/$SAVE_FOLDER/ \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \