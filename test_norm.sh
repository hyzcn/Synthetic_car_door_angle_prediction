python cnn_norm.py --command test --model-name resnet --part-name all \
                --batch-size 64 --num-images 97200 \
                --train-num 45000 --test-num 9720 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data/ \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --model-dir params/{}_ft_{}_texture.pkl \
                --plot-dir plots/{}_ft_{}_texture.jpg \
                --output-dir outputs/{}_ft_{}_texture.txt \
                --train-name-dir ImageSets/preset_all_train_norm.txt \
                --test-name-dir ImageSets/preset_all_test_norm.txt \
                --test-spatial False
# when not using test-spatial, it means using test mode, and the data dir is the same as train dir
# while using the test-spatial, the data dir is test dir