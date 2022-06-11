import cv2

cap = cv2.VideoCapture(0)  # 打开摄像头0 / 若为“xxxx.mp4”则为视频文件
isOpened = cap.isOpened()  # 判断是否打开/ 为后续提供条件语句
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)

i = 1000
while isOpened:
    if i == 2000:
        break
    else:
        i += 1
    (flag, frame) = cap.read()  # flag是否读取成功, frame为图片内容
    fileName = "image" + str(i) + ".jpg"
    if flag:
        cv2.imwrite(f"../03_DataSet/01_Grocery/{fileName}", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.imshow("Hello!", frame)
    if cv2.waitKey(1) == ord("q"):
        break

print("END")
