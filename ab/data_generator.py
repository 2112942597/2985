#if you load train dataset and valid dataset     you should use"  AUGMENTATIONS_TRAIN"
#if you load test dataset    you should use" AUGMENTATIONS_TEST"

#If you want to get the detailed code, please contact lwf@haut.edu.cn

def getData(X_shape, flag="test"):
    im_array = []
    mask_array = []

    if flag == "test":

        for i in (testing_files):
            im = cv2.resize(cv2.imread(os.path.join(image_path, i)), (X_shape, X_shape))  # [:,:,0]
            ma_ = Image.open(os.path.join(mask_path, i))




