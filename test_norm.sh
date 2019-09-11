python cnn_norm.py --command test --model-name resnet --part-name all \
                --batch-size 64 --num-images 97200 \
                --train-num 45000 --test-num 972 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data/ \
                --test-dir datasets/all_test/preset_all_random/ \
                --model-dir params/{}_ft_{}_texture.pkl \
                --plot-dir plots/{}_ft_{}_texture.jpg \
                --output-dir outputs/{}_ft_{}_texture.txt \
                --test-spatial False