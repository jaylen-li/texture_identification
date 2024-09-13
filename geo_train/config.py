# some training parameters
EPOCHS = 300 #500
BATCH_SIZE = 128 #128
NUM_CLASSES = 5
image_height = 32
image_width = 32
channels = 3

model_save_name = "EfficientNetB0" #"Xception"
model_dir = "trained_models/geospatial/"+model_save_name+"/" # = save_dir  

dir_base = "D:/project_geo/code_test/cropped_patches_35/train"
train_dir = "D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/set_{set}/train"
valid_dir = "D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/set_{set}/val"
test_dir = "D:/project_geo/code_test_combined/5folds_with_test_updated_combined_ver3/test"

test_image_path = r"D:\project_geo\code_test_changed_stride\5folds_with_test_updated1\test\Vegetation\tile1_003_vegetation_x608_y160.png"

