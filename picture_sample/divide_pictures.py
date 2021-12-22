# blog_picturesから求めるサイズの写真を取得 -> 写真を表示して手動でcorrect,incorrectに分ける
# correctは1, incorrectは0

import os
import shutil
import cv2
import matplotlib.pyplot as plt

dir_name = 'blog_pictures'
picture_names = os.listdir(dir_name)
pic_size = (1280, 960)

incorrect_list = []
correct_list = []

input('PUT ENTER')
for i in range(len(picture_names)):
    pic_path = dir_name + '/' + picture_names[i]
    pic = cv2.imread(pic_path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    if pic.shape[0:2] == pic_size:
        plt.imshow(pic)
        plt.pause(1)
        zero_one = input()
        while True:
            try:
                if int(zero_one) == 0:
                    incorrect_list.append(pic_path)
                    break
                elif int(zero_one) == 1:
                    correct_list.append(pic_path)
                    break
                else:
                    print('PUT 0 OR 1')
                    zero_one = input()

            except ValueError:
                print('PUT 0 OR 1')
                zero_one = input()

    if len(incorrect_list) == 20:
        for path in incorrect_list:
            shutil.move(path, 'dataset/0_incorrect')
            incorrect_list = []

    if len(correct_list) == 20:
        for path in correct_list:
            shutil.move(path, 'dataset/1_correct')
            correct_list = []
