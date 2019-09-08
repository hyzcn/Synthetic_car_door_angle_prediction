SAVE_FOLDER="test"
python cnn_norm.py --command test --model-name resnet --part-name all \
                --batch-size 64 --num-epoch 2 --num-images 97200 \
                --train-num 45000 --test-num 972 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data/ \
                --test-dir datasets/all_test/preset_all_random/ \
                --model-dir params/$SAVE_FOLDER/{}_ft_{}.pkl \
                --plot-dir plots/$SAVE_FOLDER/{}_ft_{}.jpg \
                --output-dir outputs/$SAVE_FOLDER/{}_ft_{}.txt \
                --html-dir htmls/$SAVE_FOLDER/{}_ft_{}.txt \
                --test-spatial False