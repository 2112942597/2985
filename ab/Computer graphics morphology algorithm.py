a = model.predict(bs)[0].squeeze()
a = np.round(a > threshold_best)
a = a.astype(np.uint8)
a = label(a)
areas = [r.area for r in regionprops(a)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(a):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                a[coordinates[0], coordinates[1]] = 0
a = a > 0
a = a.astype(np.uint8)
im_floodfill = a.copy()
h, w = a.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
isbreak = False
for i in range(im_floodfill.shape[0]):
    for j in range(im_floodfill.shape[1]):
        if (im_floodfill[i][j] == 0):
            seedPoint = (i, j)
            isbreak = True
            break
    if (isbreak):
        break
cv2.floodFill(im_floodfill, mask, seedPoint, 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = a | im_floodfill_inv
a = im_out > 0
a = a.astype(np.uint8)
a = np.where(a > 0, 255, 0)