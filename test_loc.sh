SAVE_FOLDER="loc_pre_texture"
# only when SAVE_FOLDER is locaion, use False
python cnn_loc.py --command test --model-name resnet --part-name all \
                --batch-size 64 --add-pre True \
                --feature-extract False --data-range 60 --num-images 97200 \
                --train-dir datasets/train/preset_car_data/ \
                --train-gt-dir gt_dict/preset_car_{}_gt.npy \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --test-gt-dir gt_dict/shapenet_test_all_{}_gt.npy \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}_0.3_0.6_64.pkl \
                --plot-dir plots/$SAVE_FOLDER/ \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --train-name-dir ImageSets/preset_texture_all_train.txt \
                --test-name-dir ImageSets/preset_texture_all_test.txt \
                --test-baseline True \
                --test-texture True \
                --texture False