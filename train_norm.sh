CUDA_VISIBLE_DEVICES=0 python cnn_norm.py --command train --model-name resnet --part-name all \
                --batch-size 8 --num-epoch 100 --num-images 97200 \
                --train-num 45000 --test-num 9720 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data_texture/ \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --model-dir params/{}_ft_{}_texture.pkl \
                --plot-dir plots/{}_ft_{}_texture.jpg \
                --output-dir outputs/{}_ft_{}_texture.txt \
                --test-spatial True