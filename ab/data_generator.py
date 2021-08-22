#if you load train dataset and valid dataset     you should use"  AUGMENTATIONS_TRAIN"
#if you load test dataset    you should use" AUGMENTATIONS_TEST"

def getData(X_shape, flag="test"):
    im_array = []
    mask_array = []

    if flag == "test":

        for i in (testing_files):
            im = cv2.resize(cv2.imread(os.path.join(image_path, i)), (X_shape, X_shape))  # [:,:,0]
            ma_ = Image.open(os.path.join(mask_path, i))

            ma = ma_.resize((256, 256), Image.ANTIALIAS)
            ma = np.array(ma)
            ma[ma > 0] = 1
            augmented = AUGMENTATIONS_TRAIN(image=im, mask=ma)
            image_padded = augmented['image']
            mask_padded = augmented['mask']

            im_array.append(image_padded)
            mask_array.append(mask_padded / 1.0)  # /255.0

        return im_array, mask_array



    if flag == "JSRT":
        for i in (JSRT_files_):
            im = cv2.resize(cv2.imread(os.path.join(JSRT_path, i)), (X_shape, X_shape))  # [:,:,0]

            ma = Image.open(os.path.join(JSRT_path_mask, i.split(".png")[0] + ".gif"))
            ma = ma.resize((256, 256), Image.ANTIALIAS)
            ma = np.array(ma)

            ma[ma > 0] = 1

            augmented = AUGMENTATIONS_TRAIN(image=im, mask=ma)
            image_padded = augmented['image']
            mask_padded = augmented['mask']
            im_array.append(image_padded)
            mask_array.append(mask_padded / 1.0)  # /255.0

        return im_array, mask_array

    if flag == "HAUT":
        for i in (haut_files_):
            im = cv2.resize(cv2.imread(os.path.join(haut_path, i)), (X_shape, X_shape))  # [:,:,0]
            ma = cv2.resize(cv2.imread(os.path.join(haut_path_mask, i)), (X_shape, X_shape))[:, :, 0]
            ma[ma > 0] = 1

            augmented = AUGMENTATIONS_TRAIN(image=im, mask=ma)
            image_padded = augmented['image']
            mask_padded = augmented['mask']
            im_array.append(image_padded)
            mask_array.append(mask_padded / 1.0)

        return im_array, mask_array
    dim = 256
    MC,MC_MASK = getData(dim,flag="MC")
    JSRT, JSRT_MASK = getData(dim, flag="JSRT")
    HAUT, HAUT_MASK = getData(dim, flag="HAUT")
    x_data = HAUT[:1000]
    y_data = HAUT_MASK[:1000]
    x_data = np.array(x_data).reshape(len(x_data), dim, dim, 3)
    y_data = np.array(y_data).reshape(len(y_data), dim, dim, 1)



