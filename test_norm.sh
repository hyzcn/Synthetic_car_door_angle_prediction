python cnn_norm.py --command test --model-name resnet --part-name all \
                --batch-size 64 --num-images 97200 \
                --train-num 45000 --test-num 9720 \
                --feature-extract False --data-range 60 \
                --data-dir datasets/train/preset_car_data/ \
                --train-gt-dir gt_dict/preset_car_all_gt.npy \
                --test-dir datasets/all_test/shapenet_test_all/ \
                --test-gt-dir gt_dict/shapenet_test_all_all_gt.npy \
                --model-dir params/{}_ft_{}_floor_norm_new.pkl \
                --plot-dir plots/{}_ft_{}_floor_norm_new.jpg \
                --output-dir outputs/{}_ft_{}_floor_norm_new.txt \
                --html-dir htmls/{}_ft_{}_floor_norm_new.txt \
                --train-name-dir ImageSets/preset_all_train_final_norm.txt \
                --test-name-dir ImageSets/preset_all_test_final_norm.txt \
                --test-spatial True

# when not using test-spatial, it means using test mode, and the data dir is the same as train dir
# while using the test-spatial, the data dir is test dir


# python cnn_norm_seg.py --command test --model-name resnet --part-name all \
#                 --batch-size 64 --num-images 97200 \
#                 --train-num 45000 --test-num 9720 --seg-classes 9 \
#                 --feature-extract False --data-range 60 \
#                 --data-dir datasets/train/preset_car_data/ \
#                 --train-gt-dir gt_dict/preset_car_all_gt.npy \
#                 --train-seg-gt-dir segsets/train/preset_car_data/ \
#                 --test-dir datasets/all_test/preset_test_texture/ \
#                 --test-gt-dir gt_dict/preset_test_coco_all_all_gt.npy \
#                 --test-seg-gt-dir segsets/all_test/preset_test_coco/ \
#                 --model-dir params/{}_ft_{}_floor_norm_new_seg_0.5.pkl \
#                 --plot-dir plots/{}_ft_{}_floor_norm_new_seg_0.5.jpg \
#                 --output-dir outputs/{}_ft_{}_floor_norm_new_seg_0.5.txt \
#                 --html-dir htmls/{}_ft_{}_floor_norm_new_seg_0.5.txt \
#                 --train-name-dir ImageSets/preset_all_train_final_norm.txt \
#                 --test-name-dir ImageSets/preset_all_test_final_norm.txt \
#                 --test-spatial True
