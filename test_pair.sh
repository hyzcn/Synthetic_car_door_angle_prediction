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
                --test-baseline True