import cv2
import numpy as np
from matplotlib import pyplot as plt

# 定数定義
ORG_WINDOW_NAME = "org"
GRAY_WINDOW_NAME = "gray"
CANNY_WINDOW_NAME = "canny"

ORG_FILE_NAME = "data/isojin.jpg"
GRAY_FILE_NAME = "gray.jpg"
CANNY_FILE_NAME = "canny.jpg"

# 元の画像を読み込む
org_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_UNCHANGED)
#org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)

#背景画像の読み込み
back_img = cv2.imread("data/background2.jpg", cv2.IMREAD_UNCHANGED)

#画像のサイズを合わせる
height,width =[800,1100]
print(height,width)

org_img = cv2.resize(org_img, dsize=(width,height))
back_img = cv2.resize(back_img, dsize=(width,height))
height,width = org_img.shape[:2]
print(height,width)

        
# グレースケールに変換
gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

# フィルターによる平滑化処理
for i in range(10):
    gray_img = cv2.bilateralFilter(gray_img, 15, 20, 20)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2BGR)

# エッジ抽出
canny_img = cv2.Canny(gray_img, 50, 110)

# ウィンドウに表示

fig = plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title(ORG_WINDOW_NAME)
plt.imshow(org_img)
plt.subplot(122)
plt.title(GRAY_WINDOW_NAME)
plt.imshow(gray_img)

fig = plt.figure(figsize=(16,9))
plt.title(CANNY_WINDOW_NAME)
plt.imshow(canny_img)

plt.show()


kernel = np.ones((200,200),np.uint8)
mask = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
imv_mask=cv2.bitwise_not(mask)

fig = plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title("maskedimg")
plt.imshow(mask)
plt.subplot(122)
plt.title("inv_maskedimg")
plt.imshow(imv_mask)
plt.show()


trimed_img = cv2.bitwise_and(org_img, org_img, mask=mask)
fig = plt.figure(figsize=(16,9))


plt.title("trimed_img")
plt.imshow(trimed_img)
plt.show()

#合成用の背景作成
back_img = cv2.bitwise_and(back_img, back_img,mask=imv_mask)


fig = plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title("back_img")
plt.imshow(back_img)
plt.show()


#trimed_img = cv2.resize(trimed_img, dsize=(width,height))
back_img_output = cv2.cvtColor(back_img, cv2.COLOR_RGB2BGR)
overlayed_img = np.minimum( trimed_img+back_img_output, 255).astype(np.uint8)

fig = plt.figure(figsize=(16,9))
plt.title("fusion_img")
plt.imshow(overlayed_img)
plt.show()
