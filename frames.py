import cv2

def images_per_second(video_file, output_dir, fps):
    cap = cv2.VideoCapture(video_file)
    frame_num = 0
    image_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_num % (fps // 30) == 0:
                cv2.imwrite(output_dir+"/image{}.jpg".format(image_num),frame)
                image_num += 1
        else:
            break
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    return

#images_per_second("data/clip6.mp4", "OUTPUT",30)
