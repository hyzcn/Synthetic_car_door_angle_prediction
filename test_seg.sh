SAVE_FOLDER="loc_pre_seg_0.01"
# only when SAVE_FOLDER is locaion, use False
python cnn_seg.py --command test --model-name resnet --part-name all \
                --batch-size 64 --add-pre True \
                --feature-extract False --data-range 60 --num-images 97200 \
                --train-dir datasets/train/preset_car_data/ \
                --train-gt-dir gt_dict/preset_car_{}_gt.npy \
                --train-seg-gt-dir segsets/train/preset_car_data/ \
                --test-dir datasets/all_test/preset_all_same/ \
                --test-gt-dir gt_dict/preset_all_same_{}_gt.npy \
                --test-seg-gt-dir segsets/all_test/preset_all_same/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_0.3_0.6_64.pkl \
                --plot-dir plots/$SAVE_FOLDER/ \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --train-name-dir ImageSets/preset_all_train_sample.txt \
                --test-name-dir ImageSets/preset_all_test_final_norm.txt \
                --test-texture True

# if test-baseline and test-texture, the data dir is the same as train dir
# test-texture meams read names from file
# else, the data dir is test dir