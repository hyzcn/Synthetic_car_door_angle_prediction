CUDA_VISIBLE_DEVICES=1,2 python cnn_norm.py --command train --model-name resnet --part-name all \
                --batch-size 16 --num-epoch 50 --num-images 97200 \
                --train-num 45000 --test-num 9720 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data/ \
                --test-dir datasets/all_test/preset_all_same/ \
                --model-dir params/{}_ft_{}_floor_norm_new.pkl \
                --plot-dir plots/{}_ft_{}_floor_norm_new.jpg \
                --output-dir outputs/{}_ft_{}_floor_norm_new.txt \
                --html-dir htmls/{}_ft_{}_floor_norm_new.txt \
                --train-gt-dir gt_dict/preset_car_all_gt.npy \
                --test-gt-dir gt_dict/preset_car_all_gt.npy \
                --train-name-dir ImageSets/preset_all_train_final_norm.txt \
                --test-name-dir ImageSets/preset_all_test_final_norm.txt \
                --test-spatial False > params/log_floor_new

# remember to modify saving model mode when doing different experiments!!!!!!!