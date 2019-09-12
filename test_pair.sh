SAVE_FOLDER="sigmoid_texture"
python cnn_pair.py --command test --model-name resnet --part-name all \
                --batch-size 32 --num-images 97200 \
                --feature-extract False --data-range 60 \
                --train-dir datasets/train/preset_car_data/ \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --train-name-dir ImageSets/preset_texture_all_train.txt \
                --test-name-dir ImageSets/preset_texture_all_test.txt \
                --test-baseline True \
                --test-texture False

# if test-baseline and test-texture, the data dir is the same as train dir
# test-texture meams read names from file
# else, the data dir is test dir