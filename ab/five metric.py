def accuracy_(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_positive = np.round(y_pred > threshold_best)
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = sum(y_positive * y_pred_positive)
    TN = sum(y_negative * y_pred_negative)

    FP = sum(y_negative * y_pred_positive)
    FN = sum(y_positive * y_pred_negative)

    return (TP + TN) / (TP + TN + FP + FN)


def dice_(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_positive = np.round(y_pred > threshold_best)
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = sum(y_positive * y_pred_positive)
    TN = sum(y_negative * y_pred_negative)

    FP = sum(y_negative * y_pred_positive)
    FN = sum(y_positive * y_pred_negative)

    return (2 * TP) / (2 * TP + FP + FN)


def jaccard_(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_positive = np.round(y_pred > threshold_best)
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = sum(y_positive * y_pred_positive)
    TN = sum(y_negative * y_pred_negative)

    FP = sum(y_negative * y_pred_positive)
    FN = sum(y_positive * y_pred_negative)

    return (TP) / (TP + FP + FN)


def specificity_(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_positive = np.round(y_pred > threshold_best)
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = sum(y_positive * y_pred_positive)
    TN = sum(y_negative * y_pred_negative)

    FP = sum(y_negative * y_pred_positive)
    FN = sum(y_positive * y_pred_negative)

    return (TN) / (TN + FP)


def sensitivity_(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_positive = np.round(y_pred > threshold_best)
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = sum(y_positive * y_pred_positive)
    TN = sum(y_negative * y_pred_negative)

    FP = sum(y_negative * y_pred_positive)
    FN = sum(y_positive * y_pred_negative)

    return (TP) / (TP + FN)