from ast import While
import os
# from click import command
# import cv2
import sys
import time
import socket
import ctypes
import getpass
import threading
from _thread import *
from time import strftime
from datetime import datetime
from matplotlib.pyplot import text
import numpy as np
from numpy.core.records import array
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from PIL import Image, ImageOps, ImageDraw
import numpy
from scipy.ndimage import morphology, label

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import shutil

def thread1_main(event):
    
    root = Tk()
    
    #################################################################################
    # [Initial] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Get UserName
    user_Name = getpass.getuser()

    # Path
    base_path = F"C:/Users/{user_Name}/Documents/Develop/220708_Keras_Ibutsu/"
    model_path = base_path + F"Data/keras_cnn/model/"
    image_path = base_path + F"Data/image/"
    capture_path = base_path + F"Data/keras_cnn/cnn_image/mkh_connector/3_main/1_capture/"
    binary_path = base_path + F"Data/keras_cnn/cnn_image/mkh_connector/3_main/2_result/1_binary/"
    cropped_path = base_path + F"Data/keras_cnn/cnn_image/mkh_connector/3_main/2_result/2_cropped/"
    copy_path = base_path + F"Data/keras_cnn/cnn_image/mkh_connector/3_main/2_result/3_judge/"
    cap_original_path = capture_path + F"original/"
    cap_binary_path = capture_path + F"binary/"

    

    # Path : image
    ng_red = image_path + F"NG_RED.png"
    ng_yellow = image_path + F"NG_YELLOW.png"
    ok_green = image_path + F"OK.png"
    warning_red = image_path + F"WARNING_RED.png"
    warning_yellow = image_path + F"WARNING_YELLOW.png"
    warning_default = image_path + F"WARNING_DEFAULT.png"
    no_image = image_path + F"NO_IMAGE_MAIN.png"
    no_image_crop = image_path + F"NO_IMAGE_CROP.png"

    
    # Display Size        
    user32 = ctypes.windll.user32
    display_Size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    wm_attr = "#000"
    bg_Color = "#161931"
    dp_w, dp_h = display_Size[0], display_Size[1]
    default_DpSize = [1920, 1080]
    dp_size = [dp_w, dp_h, 0, 0] # width, height, x start, y start

    # Title
    root.geometry(F"{dp_size[0]}x{dp_size[1]}+{dp_size[2]}+{dp_size[3]}")
    root.wm_attributes('-transparentcolor', wm_attr)
    root.configure(background=bg_Color)
    root.title("MKH")
    root.overrideredirect(True)


    #################################################################################
    # [Function] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def tk_image(image_path):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
        image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
        return image_tk

    def btn1_capture(cap_area_w, cap_area_h, crop_area_w, crop_area_h):
        L9_top.configure(bg="#FC0E73")
        L11_top.configure(bg=bg_Color)
        L13_top.configure(bg=bg_Color)
        L15_top.configure(bg=bg_Color)

        # YYMMDD_HHMMSS : 사진파일 이름에 들어갈 날짜 
        now_time = datetime.today().strftime("%Y%m%d_%H%M%S")[2:]
        # 함수 실행하자 찍힌 초(second)
        in_second = datetime.today().strftime("%S")

        # PATH
        original_img = cap_original_path + F"original.{now_time}.jpg"
        binary_img = cap_binary_path + F"binary.{now_time}.jpg"

        # 카메라 설정(해상도 1280x1024)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

        # Wating Camera Flash : MKH
        while True:
            ret, frame = cap.read()
            now_second = datetime.today().strftime("%S")
            delay_flash = abs(int(in_second) - int(now_second))
            print(delay_flash)
            if delay_flash >= 9:      
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, dst = cv2.threshold(gray, 152, 255, cv2.THRESH_BINARY) # Home : 127, 255 / MKH : 152, 255
                
                cv2.imwrite(original_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(binary_img, dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
                break

        cap.release()
        cv2.destroyAllWindows()

        # 각각의 사진이(원본, 바이너리) 저장된 폴더 경로
        original_file_list = os.listdir(cap_original_path)
        binary_file_list = os.listdir(cap_binary_path)
        
        
        # 각각의 경로에서 가장 나중에 찍은(가장 최신의) 사진파일
        original_capture_file = cap_original_path + original_file_list[len(original_file_list)-1]
        binary_capture_file = cap_binary_path + binary_file_list[len(binary_file_list)-1]
        
        # Update Image
        capture_image = Image.open(original_capture_file)
        cap_img = capture_image.resize((cap_area_w, cap_area_h))
        cap_img = ImageTk.PhotoImage(cap_img)
        c1_capture_label.configure(image=cap_img)
        c1_capture_label.image = cap_img


        # Turn 2


        # 바이너리 파일을 줄테니 원을 가져오너라
        def boxes(binary_img):
            img = ImageOps.grayscale(binary_img)
            im = numpy.array(img)

            # Inner morphological gradient.
            im = morphology.grey_dilation(im, (3, 3)) - im

            # Binarize.
            mean, std = im.mean(), im.std()
            t = mean + std
            im[im < t] = 0
            im[im >= t] = 1

            # Connected components.
            lbl, numcc = label(im)
            # Size threshold.
            min_size = 200 # pixels
            box = []
            for i in range(1, numcc + 1):
                py, px = numpy.nonzero(lbl == i)
                if len(py) < min_size:
                    im[lbl == i] = 0
                    continue

                xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
                # Four corners and centroid.
                box.append([
                    [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
                    (numpy.mean(px), numpy.mean(py))])

            return im.astype(numpy.uint8) * 255, box

        # YYMMDD_HHMMSS : 사진파일 이름에 들어갈 날짜 
        now_time = datetime.today().strftime("%Y%m%d_%H%M%S")[2:]

        # 카메라에서 가장 나중에 찍었던 사진(오리지널, 바이너리)
        original_img = Image.open(original_capture_file)
        binary_img = Image.open(binary_capture_file)

        # 원형 검출 시 검출해야 할 사이즈
        # - 바이너리 파일에서 원이라고 판단되는 구역을 찾을 때
        #   해당 원 사이즈를 미리 설정
        # - 이걸 안 해놓으면 원이 아닌 것도 원이라고 잡는 경우가 있다
        # - 사진 해상도는 1280x1024 기준이다
        min_size, max_size = 125, 165

        # 바이너리 파일을 줄테니 원을 가져오너라
        im, box = boxes(binary_img)

        # Draw perfect rectangles and the component centroid.
        img = Image.fromarray(im)
        visual = img.convert('RGB')
        draw = ImageDraw.Draw(visual)

        # 이 부분에서는 바이너리 사진파일 안에 있는 모든 요소를 찾는 것 같다
        for b, centroid in box:
            r_point = b + [b[0]]
            
            # 해당 오브젝트를 검출한 포인트의 길이(11시 방향, 1시 방향, 5시 방향, 7시 방향)
            # - point_11 : 11시 방향 ~ 1시 방향까지의 길이
            # - point_1 : 1시 방향 ~ 5시 방향까지의 길이
            # - point_5 : 5시 방향 ~ 7시 방향까지의 길이
            # - point_7 : 7시 방향 ~ 11시 방향까지의 길이
            point_11 = int(r_point[1][0] - r_point[0][0])
            point_1 = int(r_point[2][1] - r_point[1][1])
            point_5 = int(r_point[2][0] - r_point[3][0])
            point_7 = int(r_point[3][1] - r_point[4][1])

            print(point_11, point_1, point_5, point_7)

            # 각 포인트의 길이가 미리 설정해놓은 min_size, max_size(140, 170) 이하인가
            if (point_11 >= min_size and point_11 <= max_size) and (point_1 >= min_size and point_1 <= max_size) \
                and (point_5 >= min_size and point_5 <= max_size) and (point_7 >= min_size and point_7 <= max_size):
                
                # 검출한 포인트의 비율을 구함
                # - ave_top_right : 11시 방향 ~ 1시 방향까지의 길이 /  1시 방향 ~ 5시 방향까지의 길이의 비율
                # - ave_right_bottom : 1시 방향 ~ 5시 방향까지의 길이 /  5시 방향 ~ 7시 방향까지의 길이의 비율
                # - ave_bottom_left : 5시 방향 ~ 7시 방향까지의 길이 /  7시 방향 ~ 11시 방향까지의 길이의 비율
                # - ave_left_top : 7시 방향 ~ 11시 방향까지의 길이 /  11시 방향 ~ 1시 방향까지의 길이의 비율
                ave_top_right = round(point_11 / point_1, 0)
                ave_right_bottom = round(point_1 / point_5, 0)
                ave_bottom_left = round(point_5 / point_7, 0)
                ave_left_top = round(point_7 / point_11, 0)

                # 각각의 포인트의 비율이 1.0인가(정사각형인가)
                # - 정사각형 이외의 비율은 컷
                if ave_top_right == 1.0 and ave_right_bottom == 1.0 and ave_bottom_left == 1.0 and ave_left_top == 1.0:
                    
                    # 조건에 맞는 곳의 좌표
                    # left / top / right / bottom : 11방향 / 1시방향 / 5시방향 / 7시방향
                    left = int(r_point[0][0])
                    top = int(r_point[0][1])
                    right = int(r_point[2][0])
                    bottom = int(r_point[2][1])

                    # 좌표의 이미지를 크롭
                    # - 정상적으로 크롭되었다면 부품 내 가운데 동그라미가 크롭되어야 한다
                    im1 = original_img.crop((left, top, right, bottom))
                    im1.save(cropped_path + F"{now_time}_[{left}.{top}.{right}.{bottom}].jpg", quality=95)
                    
                    # 해당 부분을 좌표를 빨간색 선으로 이어서 표시
                    draw.line(b + [b[0]], fill='red')

            cx, cy = centroid
            # 해당 부분을 좌표의 중심지점을 노란색 점으로 표시
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='yellow')

        # 검출 결과의 사진파일을 저장한다
        visual.save(binary_path + F"{now_time}.jpg")



        cropped_file_list = os.listdir(cropped_path)
        crop_right_file = cropped_path + cropped_file_list[len(cropped_file_list)-1]
        crop_left_file = cropped_path + cropped_file_list[len(cropped_file_list)-2]

        # Ture 3?

        image_size = (100, 100)
        batch_size = 32

        left_circle_safe_per, right_circle_safe_per = 0, 0

        def copy_cropped_image(crop_file, judge_flag):
            if os.path.exists(crop_file):
                if judge_flag == True:
                    shutil.copy2(crop_file, copy_path + F"ok/")
                elif judge_flag == False:
                    shutil.copy2(crop_file, copy_path + F"ng/")
                else:
                    print("Copy Error")

        for i in range(1,3):
            if i == 1:
                img = tf.keras.preprocessing.image.load_img(
                    crop_left_file, target_size=image_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis

                model = load_model(model_path + "model_MKH_722.h5")
                # model = load_model("save_at_50.h5")
                predictions = model.predict(img_array)
                score = predictions[0]
                # print(
                #     "This image is %.2f percent ng and %.2f percent safe."
                #     % (100 * (1 - score), 100 * score)
                # )
                left_circle_safe_per = 100 * score
                left_circle_safe_per = int(left_circle_safe_per)

                if left_circle_safe_per >= 80 and left_circle_safe_per <= 100:
                    print(left_circle_safe_per, "ok")
                    copy_cropped_image(crop_left_file, True)
                elif left_circle_safe_per >= 0 and left_circle_safe_per < 80:
                    print(left_circle_safe_per, "ng")
                    copy_cropped_image(crop_left_file, False)
                else:
                    print("error")


            if i == 2:      
                img = tf.keras.preprocessing.image.load_img(
                    crop_right_file, target_size=image_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis

                model = load_model(model_path + "save_at_50.h5")
                # model = load_model("save_at_50.h5")
                predictions = model.predict(img_array)
                score = predictions[0]
                # print(
                #     "This image is %.2f percent ng and %.2f percent safe."
                #     % (100 * (1 - score), 100 * score)
                # )
                right_circle_safe_per = 100 * score
                right_circle_safe_per = int(right_circle_safe_per)

                if right_circle_safe_per >= 80 and right_circle_safe_per <= 100:
                    print(right_circle_safe_per, "ok")
                    copy_cropped_image(crop_right_file, True)
                elif right_circle_safe_per >= 0 and right_circle_safe_per < 80:
                    print(right_circle_safe_per, "ng")
                    copy_cropped_image(crop_right_file, False)
                else:
                    print("error")

        
        # Update Image
        crop_left_image = Image.open(crop_left_file)
        crop_left_img = crop_left_image.resize((crop_area_w, crop_area_h))
        crop_left = ImageTk.PhotoImage(crop_left_img)
        c2_crop_image_left_label.configure(image=crop_left)
        c2_crop_image_left_label.image = crop_left

        crop_right_file = Image.open(crop_right_file)
        crop_right_img = crop_right_file.resize((crop_area_w, crop_area_h))
        crop_right = ImageTk.PhotoImage(crop_right_img)
        c3_crop_image_right_label.configure(image=crop_right)
        c3_crop_image_right_label.image = crop_right

        
        
        print(left_circle_safe_per, right_circle_safe_per)

        if left_circle_safe_per >= 80:
            L2_r_coincidence_right.configure(text=F"致率：{left_circle_safe_per}%", bg="#B5E61D")
        else:
            L2_r_coincidence_right.configure(text=F"合致率：{left_circle_safe_per}%", bg="#f00")
        
        if right_circle_safe_per >= 80:
            L4_r_coincidence_right.configure(text=F"合致率：{right_circle_safe_per}%", bg="#B5E61D")
        else:
            L4_r_coincidence_right.configure(text=F"合致率：{right_circle_safe_per}%", bg="#f00")



        if left_circle_safe_per >= 80 and right_circle_safe_per >= 80:
            ok_green_image = Image.open(ok_green)
            ok_green_img = ok_green_image.resize((478, 387))
            green_img = ImageTk.PhotoImage(ok_green_img)
            c4_result_label.configure(image=green_img)
            c4_result_label.image = green_img

            warning_default_image = Image.open(warning_default)
            warning_default_img = warning_default_image.resize((478, 258))
            default_img = ImageTk.PhotoImage(warning_default_img)
            c5_warning_message_label.configure(image=default_img)
            c5_warning_message_label.image = default_img
        
        else:
            ng_yellow_image = Image.open(ng_yellow)
            ng_yellow_img = ng_yellow_image.resize((478, 387))
            yellow_img = ImageTk.PhotoImage(ng_yellow_img)
            c4_result_label.configure(image=yellow_img)
            c4_result_label.image = yellow_img

            warning_yellow_image = Image.open(warning_yellow)
            warning_yellow_img = warning_yellow_image.resize((478, 258))
            warning_img = ImageTk.PhotoImage(warning_yellow_img)
            c5_warning_message_label.configure(image=warning_img)
            c5_warning_message_label.image = warning_img





    def btn2_save():
        L9_top.configure(bg=bg_Color)
        L11_top.configure(bg="#FC0E73")
        L13_top.configure(bg=bg_Color)
        L15_top.configure(bg=bg_Color)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:    
            ret, frame = cap.read()
            now_second = datetime.today().strftime("%S")

            cv2.imshow("Focus Setting", frame)
            cv2.moveWindow("Focus Setting", 170,150)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


        

    def btn3_maintenance():
        L9_top.configure(bg=bg_Color)
        L11_top.configure(bg=bg_Color)
        L13_top.configure(bg="#FC0E73")
        L15_top.configure(bg=bg_Color)

    def btn4_exit():
        L9_top.configure(bg=bg_Color)
        L11_top.configure(bg=bg_Color)
        L13_top.configure(bg=bg_Color)
        L15_top.configure(bg="#FC0E73")

        
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            global shutdown_flag
            shutdown_flag = 1
            sys.exit()
        
    #################################################################################
    # [Drawing] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # Shutdown
    global shutdown_flag
    shutdown_flag = 0

    # Split bar (when Capture image to Cropped image) 
    bar_size = 10

    # Button Event
    btn_cur, btn_rel = "hand2", "raised"

    no_img = ImageTk.PhotoImage(Image.open(no_image))
  
    # Canvas_1 : Camera Area / 54, 54, 900, 720
    c1_bg = "#111"
    c1_x, c1_y = int(dp_h * 0.05), int(dp_h * 0.05)
    c1_w, c1_h = int(dp_w * 0.667), int(dp_h * 0.667)
    c1_w = int((c1_h / 4) * 5) # 1280x1024(5:4) = 900x720(5:4)
    c1_capture = Canvas(root, highlightthickness=0)
    c1_capture.place(x=c1_x, y=c1_y, w=c1_w, h=c1_h)
    
    c1_capture_label = Label(c1_capture, image=no_img)
    c1_capture_label.pack(side="bottom", fill="both", expand="yes")

    # Label_1 : Split Bar Vertical / 954, 54, 10, 720
    L1_bg, L1_fg = bg_Color, "#fff"
    L1_x, L1_y = c1_x+c1_w, c1_y
    L1_w, L1_h = bar_size, c1_h

    L1_split_bar_vertical = Label(root, bg=L1_bg, fg=L1_fg)
    L1_split_bar_vertical.place(x=L1_x, y=L1_y, w=L1_w, h=L1_h)

    # Canvas_2 : Cropped Image(Left) / 964, 54, 370, 284
    c2_x, c2_y = c1_x + c1_w + L1_w, c1_y
    c2_w, c2_h = int(dp_w * 0.667) - (c1_w + L1_w), int((c1_h - L1_w) / 2 * 0.8)
    
    c2_crop_image_left = Canvas(root,  highlightthickness=0)
    c2_crop_image_left.place(x=c2_x, y=c2_y, w=c2_w, h=c2_h)

    no_img_crop = ImageTk.PhotoImage(Image.open(no_image_crop))
    
    c2_crop_image_left_label = Label(c2_crop_image_left, image=no_img_crop)
    c2_crop_image_left_label.pack(side="bottom", fill="both", expand="yes")

    # Label_2L : Left Text / 964, 338, 370, 71
    L2_l_bg, L2_l_fg = "#363d78", "#fff"
    L2_l_font, L2_l_text = ["Tahoma", 32, "bold"], "L"
    L2_l_x, L2_l_y = c2_x, c2_y + c2_h
    L2_l_w, L2_l_h = c2_w*0.2, int((c1_h - L1_w) / 2 * 0.2)

    L2_l_left_text = Label(root, font=(L2_l_font[0], L2_l_font[1], L2_l_font[2]),
                        text=L2_l_text, bg=L2_l_bg, fg=L2_l_fg)
    L2_l_left_text.place(x=L2_l_x, y=L2_l_y, w=L2_l_w, h=L2_l_h)

    # Label_2R : Display Canvas_2 Coincidence(%) / 964, 338, 370, 71
    L2_r_bg, L2_r_fg = "#262b54", "#fff"
    L2_r_font, L2_r_text = ["Tahoma", 24, "bold"], "Null"
    L2_r_x, L2_r_y = c2_x + L2_l_w, c2_y + c2_h
    L2_r_w, L2_r_h = c2_w*0.8, int((c1_h - L1_w) / 2 * 0.2)

    L2_r_coincidence_right = Label(root, font=(L2_r_font[0], L2_r_font[1], L2_r_font[2]),
                        text=L2_r_text, bg=L2_r_bg, fg=L2_r_fg)
    L2_r_coincidence_right.place(x=L2_r_x, y=L2_r_y, w=L2_r_w, h=L2_r_h)

    # Label_3 : Split Bar Horizon / 964, 409, 370, 10
    L3_bg, L3_fg = bg_Color, "#fff"
    L3_x, L3_y = c2_x, c1_y + c2_h + L2_r_h
    L3_w, L3_h = c2_w, bar_size

    L3_split_bar_horizon = Label(root, bg=L3_bg, fg=L3_fg)
    L3_split_bar_horizon.place(x=L3_x, y=L3_y, w=L3_w, h=L3_h)

    # Canvas_3 : Cropped Image(right) / 964, 419, 370, 284
    c3_x, c3_y = c2_x, L3_y + L3_h
    c3_w, c3_h = int(dp_w * 0.667) - (c1_w + L1_w), c2_h
    
    c3_crop_image_right = Canvas(root,  highlightthickness=0)
    c3_crop_image_right.place(x=c3_x, y=c3_y, w=c3_w, h=c3_h)

    c3_crop_image_right_label = Label(c3_crop_image_right, image=no_img_crop)
    c3_crop_image_right_label.pack(side="bottom", fill="both", expand="yes")

    # Label_4L : Right Text / 964, 338, 370, 71
    L4_l_bg, L4_l_fg = "#363d78", "#fff"
    L4_l_font, L4_l_text = ["Tahoma", 32, "bold"], "R"
    L4_l_x, L4_l_y = c2_x, c3_y + c3_h
    L4_l_w, L4_l_h = c2_w*0.2, int((c1_h - L1_w) / 2 * 0.2)

    L4_l_left_text = Label(root, font=(L2_l_font[0], L2_l_font[1], L2_l_font[2]),
                        text=L4_l_text, bg=L4_l_bg, fg=L4_l_fg)
    L4_l_left_text.place(x=L4_l_x, y=L4_l_y, w=L4_l_w, h=L4_l_h)

    # Label_4R : Display Canvas_2 Coincidence(%) / 964, 338, 370, 71
    L4_r_bg, L4_r_fg = "#262b54", "#fff"
    L4_r_font, L4_r_text = ["Tahoma", 24, "bold"], "Null"
    L4_r_x, L4_r_y = c2_x + L2_l_w, c3_y + c3_h
    L4_r_w, L4_r_h = c2_w*0.8, L2_r_h

    L4_r_coincidence_right = Label(root, font=(L4_r_font[0], L4_r_font[1], L4_r_font[2]),
                        text=L4_r_text, bg=L4_r_bg, fg=L4_r_fg)
    L4_r_coincidence_right.place(x=L4_r_x, y=L4_r_y, w=L4_r_w, h=L4_r_h)

    # Label_5 : Split Bar Vertical Right / 1334, 54, 54, 720
    L5_bg, L5_fg = bg_Color, "#fff"
    L5_x, L5_y = c2_x + c2_w, c2_y
    L5_w, L5_h = c1_x, c1_h

    L5_split_bar_vertical_right = Label(root, bg=L5_bg, fg=L5_fg)
    L5_split_bar_vertical_right.place(x=L5_x, y=L5_y, w=L5_w, h=L5_h)

    # # Display now time
    now_time = datetime.today().strftime("%y.%m.%d %H:%M")
    
    # Label_6 : Time(yy.mm.dd hh:mm) / 1388, 54, 478, 54
    L6_bg, L6_fg = "#000", "#fff"
    L6_x, L6_y = L5_x+L5_w, L5_y
    L6_w, L6_h = dp_w - ((c1_x*3) + c1_w + bar_size + c2_w), c1_x

    global time_label_x, time_label_y, time_label_w, time_label_h 
    time_label_x, time_label_y, time_label_w, time_label_h = L6_x, L6_y, L6_w, L6_h

    L6_time = Label(root, bg=L6_bg, fg=L6_fg)
    L6_time.place(x=L6_x, y=L6_y, w=L6_w, h=L6_h)

    # Canvas_4 : Result / 1388, 118, 478, 387
    c4_x, c4_y = L6_x, L6_y + L6_h + bar_size
    c4_w, c4_h = L6_w, (c1_h - (L6_h + (bar_size * 2))) * 0.6

    c4_result = Canvas(root, highlightthickness=0, bg=bg_Color)
    c4_result.place(x=c4_x, y=c4_y, w=c4_w, h=int(c4_h))

    c4_result_label = Label(c4_result, bg=bg_Color)
    c4_result_label.pack(side="bottom", fill="both", expand="yes")

    # Label_7 : Split Bar Horizon Right / 1388, 505, 478, 10
    L7_bg, L7_fg = bg_Color, "#fff"
    L7_x, L7_y = L6_x, c4_y + c4_h
    L7_w, L7_h = L6_w, bar_size

    L7_split_bar_horizon_right = Label(root, bg=L7_bg, fg=L7_fg)
    L7_split_bar_horizon_right.place(x=L7_x, y=L7_y, w=L7_w, h=L7_h)

    # Canvas_5 : Warning Message / 1388, 515, 478, 258
    c5_x, c5_y = L6_x, L7_y + L7_h
    c5_w, c5_h = L6_w, (c1_h - (L6_h + (bar_size * 2))) * 0.4

    c5_warning_message = Canvas(root, highlightthickness=0, bg=bg_Color)
    c5_warning_message.place(x=c5_x, y=int(c5_y), w=c5_w, h=int(c5_h))

    c5_warning_message_label = Label(c5_warning_message, bg=bg_Color)
    c5_warning_message_label.pack(side="bottom", fill="both", expand="yes")

    # img3 = tk_image(warning_yellow)
    # img2= PhotoImage(file=ng_yellow)

    # captured_image = c5_warning_message.create_image(0, 0, image=img3, anchor='nw') # ImageTk 画像配置

    # Label_8 : Split Bar Horizon Right / 54, 774, 1812, 54
    L8_bg, L8_fg = bg_Color, "#fff"
    L8_x, L8_y = c1_x, c1_y + c1_h
    L8_w, L8_h = dp_w - (c1_x * 2), c1_y

    L8_split_bar_horizon_bottom = Label(root, bg=L8_bg, fg=L8_fg)
    L8_split_bar_horizon_bottom.place(x=L8_x, y=L8_y, w=L8_w, h=L8_h)

    # Canvas_6 : Button Area / 54, 828, 1812, 198
    c6_bg = "#161931"
    c6_font = ["Tahoma", 24, "bold"]
    c6_x, c6_y = c1_x, L8_y + L8_h
    c6_w, c6_h = dp_w - (c1_x * 2), dp_h - ((c1_x * 3) + c1_h)

    c6_button_outer = Canvas(root, highlightthickness=0)
    c6_button_outer.place(x=c6_x, y=c6_y, w=c6_w, h=c6_h)

    # Canvas_7 : Capture Button Outer / 19, 19, 428, 158
    c7_bg = "#161931"
    c7_x, c7_y = c6_h * 0.1, c6_h * 0.1
    c7_w, c7_h = (c6_w - (c7_x * 5)) / 4, c6_h * 0.8

    c7_button_capture = Canvas(c6_button_outer, highlightthickness=0, bg=c7_bg)
    c7_button_capture.place(x=int(c7_x), y=int(c7_y), w=int(c7_w), h=int(c7_h))
    
    # Label_9 : Capture Button Top / 10, 7, 406, 15
    L9_bg, L9_fg = bg_Color, "#fff"
    L9_x, L9_y = c7_w * 0.025, c7_h * 0.05
    L9_w, L9_h = c7_w * 0.95, c7_h * 0.1

    L9_top = Label(c7_button_capture, bg=L9_bg, fg=L9_fg)
    L9_top.place(x=int(L9_x), y=int(L9_y), w=int(L9_w), h=int(L9_h))



    # Split bar (when Capture image to Cropped image) 
    bar_size = 10
    
    # default_image_path = "C:/Users/ROG3070/Documents/Develop/220708_Keras_Ibutsu/Data/image/NO_IMAGE_MAIN.jpg"    

    # no_image_main = tk_image(default_image_path)
    # c_no_image_main = c1_capture.create_image(0, 0, image=no_image_main, anchor='nw') # ImageTk 画像配置






    # Button_1 : Capture / 10, 31, 406, 79
    b1_bg, b1_fg = "#262b54", "#fff"
    b1_ac_bg, b1_ac_fg = "#B5E61D", b1_bg
    b1_font, b1_text = ["Tahoma", 36, "bold"], "判定"
    b1_x, b1_y = c7_w * 0.025, L9_y + L9_h + c7_h * 0.05
    b1_w, b1_h = c7_w * 0.95, c7_h * 0.5

    b1_capture = Button(c7_button_capture, font=(b1_font[0], b1_font[1], b1_font[2]),
                        text=b1_text, bg=b1_bg, fg=b1_fg, borderwidth=0, 
                        activebackground=b1_ac_bg, activeforeground=b1_ac_fg,
                        cursor=btn_cur, relief=btn_rel, command=lambda:btn1_capture(c1_w, c1_h, c2_w, c2_h))
    b1_capture.place(x=int(b1_x), y=int(b1_y), w=int(b1_w), h=int(b1_h))
    










    # Label_10 : Capture Button Bottom / 10, 118, 406, 31
    L10_bg, L10_fg = bg_Color, "#fff"
    L10_font, L10_text = ["Tahoma", 20, "bold"], "C A P T U R E"
    L10_x, L10_y = c7_w * 0.025, b1_y + b1_h + c7_h * 0.05
    L10_w, L10_h = c7_w * 0.95, c7_h * 0.2

    L10_bottom = Label(c7_button_capture, font=(L10_font[0], L10_font[1], L10_font[2]),
                        text=L10_text, bg=L10_bg, fg=L10_fg)
    L10_bottom.place(x=int(L10_x), y=int(L10_y), w=int(L10_w), h=int(L10_h))

    # Canvas_8 : Save Button Outer / 467, 19, 428, 158
    c8_bg = "#161931"
    c8_x, c8_y = (c7_x * 2) + c7_w, c7_y
    c8_w, c8_h = c7_w, c7_h

    c8_button_save = Canvas(c6_button_outer, highlightthickness=0, bg=c8_bg)
    c8_button_save.place(x=int(c8_x), y=int(c8_y), w=int(c8_w), h=int(c8_h))
    
    # Label_11 : Save Button Top / 10, 7, 406, 15
    L11_bg, L11_fg = bg_Color, "#fff"
    L11_x, L11_y = c8_w * 0.025, c8_h * 0.05
    L11_w, L11_h = c8_w * 0.95, c8_h * 0.1

    L11_top = Label(c8_button_save, bg=L11_bg, fg=L11_fg)
    L11_top.place(x=int(L11_x), y=int(L11_y), w=int(L11_w), h=int(L11_h))

    # Button_2 : Save / 10, 31, 406, 79
    b2_bg, b2_fg = "#262b54", "#fff"
    b2_text = "保存"
    b2_x, b2_y = c8_w * 0.025, L9_y + L9_h + c8_h * 0.05
    b2_w, b2_h = c8_w * 0.95, c7_h * 0.5

    b2_save = Button(c8_button_save, font=(b1_font[0], b1_font[1], b1_font[2]),
                        text=b2_text, bg=b2_bg, fg=b2_fg, borderwidth=0,
                        activebackground=b1_ac_bg, activeforeground=b1_ac_fg,
                        cursor=btn_cur, relief=btn_rel, command=btn2_save)
    b2_save.place(x=int(b2_x), y=int(b2_y), w=int(b2_w), h=int(b2_h))

    # Label_12 : Save Button Bottom / 10, 118, 406, 31
    L12_bg, L12_fg = bg_Color, "#fff"
    L12_font, L12_text = ["Tahoma", 20, "bold"], "S A V E"
    L12_x, L12_y = c8_w * 0.025, b1_y + b1_h + c8_h * 0.05
    L12_w, L12_h = c8_w * 0.95, c8_h * 0.2

    L12_bottom = Label(c8_button_save, font=(L12_font[0], L12_font[1], L12_font[2]),
                        text=L12_text, bg=L12_bg, fg=L12_fg)
    L12_bottom.place(x=int(L12_x), y=int(L12_y), w=int(L12_w), h=int(L12_h))

    # Canvas_9 : Maintenance Button Outer / 915, 19, 428, 158
    c9_bg = "#161931"
    c9_x, c9_y = (c7_x * 3) + (c7_w * 2), c7_y
    c9_w, c9_h = c7_w, c7_h

    c9_button_maintenance = Canvas(c6_button_outer, highlightthickness=0, bg=c9_bg)
    c9_button_maintenance.place(x=int(c9_x), y=int(c9_y), w=int(c9_w), h=int(c9_h))

    # Label_13 : Maintenance Button Top / 10, 7, 406, 15
    L13_bg, L13_fg = bg_Color, "#fff"
    L13_x, L13_y = c9_w * 0.025, c9_h * 0.05
    L13_w, L13_h = c9_w * 0.95, c9_h * 0.1

    L13_top = Label(c9_button_maintenance, bg=L13_bg, fg=L13_fg)
    L13_top.place(x=int(L13_x), y=int(L13_y), w=int(L13_w), h=int(L13_h))

    # Button_3 : Maintenance / 10, 31, 406, 79
    b3_bg, b3_fg = "#262b54", "#fff"
    b3_text = "学習"
    b3_x, b3_y = c9_w * 0.025, L9_y + L9_h + c9_h * 0.05
    b3_w, b3_h = c9_w * 0.95, c9_h * 0.5

    b3_maintenance = Button(c9_button_maintenance, font=(b1_font[0], b1_font[1], b1_font[2]),
                        text=b3_text, bg=b3_bg, fg=b3_fg, borderwidth=0,
                        activebackground=b1_ac_bg, activeforeground=b1_ac_fg, 
                        cursor=btn_cur, relief=btn_rel, command=btn3_maintenance)
    b3_maintenance.place(x=int(b3_x), y=int(b3_y), w=int(b3_w), h=int(b3_h))

    # Label_14 : Maintenance Button Bottom / 10, 118, 406, 31
    L14_bg, L14_fg = bg_Color, "#fff"
    L14_font, L14_text = ["Tahoma", 20, "bold"], "L E A R N I N G"
    L14_x, L14_y = c9_w * 0.025, b1_y + b1_h + c9_h * 0.05
    L14_w, L14_h = c9_w * 0.95, c9_h * 0.2

    L14_bottom = Label(c9_button_maintenance, font=(L14_font[0], L14_font[1], L14_font[2]),
                        text=L14_text, bg=L14_bg, fg=L14_fg)
    L14_bottom.place(x=int(L14_x), y=int(L14_y), w=int(L14_w), h=int(L14_h))

    # Canvas_10 : Exit Button Outer / 1363, 19, 428, 158
    c10_bg = "#161931"
    c10_x, c10_y = (c7_x * 4) + (c7_w * 3), c7_y
    c10_w, c10_h = c7_w, c7_h

    c10_button_exit = Canvas(c6_button_outer, highlightthickness=0, bg=c10_bg)
    c10_button_exit.place(x=int(c10_x), y=int(c10_y), w=int(c10_w), h=int(c10_h))

    # Label_15 : Exit Button Top / 10, 7, 406, 15
    L15_bg, L15_fg = bg_Color, "#fff"
    L15_x, L15_y = c10_w * 0.025, c10_h * 0.05
    L15_w, L15_h = c10_w * 0.95, c10_h * 0.1

    L15_top = Label(c10_button_exit, bg=L15_bg, fg=L15_fg)
    L15_top.place(x=int(L15_x), y=int(L15_y), w=int(L15_w), h=int(L15_h))

    # Button_4 : Exit / 10, 31, 406, 79
    b4_bg, b4_fg = "#262b54", "#fff"
    b4_text = "終了"
    b4_x, b4_y = c10_w * 0.025, L9_y + L9_h + c10_h * 0.05
    b4_w, b4_h = c10_w * 0.95, c10_h * 0.5

    b4_exit = Button(c10_button_exit, font=(b1_font[0], b1_font[1], b1_font[2]),
                        text=b4_text, bg=b4_bg, fg=b4_fg, borderwidth=0,
                        activebackground=b1_ac_bg, activeforeground=b1_ac_fg,
                        cursor=btn_cur, relief=btn_rel, command=btn4_exit)
    b4_exit.place(x=int(b4_x), y=int(b4_y), w=int(b4_w), h=int(b4_h))

    # Label_16 : Maintenance Button Bottom / 10, 118, 406, 31
    L16_bg, L16_fg = bg_Color, "#fff"
    L16_font, L16_text = ["Tahoma", 20, "bold"], "E X I T"
    L16_x, L16_y = c10_w * 0.025, b1_y + b1_h + c10_h * 0.05
    L16_w, L16_h = c10_w * 0.95, c10_h * 0.2

    L16_exit = Label(c10_button_exit, font=(L16_font[0], L16_font[1], L16_font[2]),
                        text=L16_text, bg=L16_bg, fg=L16_fg)
    L16_exit.place(x=int(L16_x), y=int(L16_y), w=int(L16_w), h=int(L16_h))


    



    root.mainloop()










class Display_Time:
    def __init__(self, root):
        self.root = root
        
        dp_size = [1920, 1080, 0, 0] # width, height, x start, y start
        self.wm_attr = "#000"
        root.geometry(F"{dp_size[0]}x{dp_size[1]}+{dp_size[2]}+{dp_size[3]}")
        self.root.wm_attributes('-transparentcolor', self.wm_attr)
        root.configure(background=self.wm_attr)
        root.title("Client System")
        root.overrideredirect(True)
        
        # Global not defined delay
        time.sleep(1)

        # Drawing Layout
        self.drawing_Layout()

        # Update
        self.delay_Layout = 120
        self.update_Layout()
    
    def drawing_Layout(self):

        self.L_bg, self.L_fg = "#111", "#fff"
        self.L_font = ["Tahoma", 20, "bold"]
 
        now_time = datetime.today().strftime("%y.%m.%d %H:%M")
        self.label = Label(self.root, font=(self.L_font[0], int(self.L_font[1]*1.2), self.L_font[2]),
                                    text=now_time, bg=self.L_bg, fg=self.L_fg)
        # self.label.place(x=time_label_x, y=time_label_y, w=time_label_w, h=time_label_h)
        self.label.place(x=1388, y=54, w=478, h=54)

    def update_Layout(self):
        now_time = datetime.today().strftime("%y.%m.%d %H:%M")
        self.label.configure(text=now_time)

        if shutdown_flag is not None:
            
            if shutdown_flag == 1:
                sys.exit()

        self.root.after(self.delay_Layout, self.update_Layout)


def thread2_time(event):
    root = Tk()
    obj = Display_Time(root)
    root.mainloop()

def callback(event):
    t1_main = threading.Thread(target=thread1_main, args=(event,))
    t1_main.start()

    t2_time = threading.Thread(target=thread2_time, args=(event,))
    t2_time.start()

if __name__ == "__main__":
    bind = "<1>"
    callback(bind)