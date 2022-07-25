import os
import cv2
import sys
import time
import numpy
import shutil
import random
import getpass
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import morphology, label

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 
# 
# 
def cap_origin_binary(base_path):

    # YYMMDD_HHMMSS : 사진파일 이름에 들어갈 날짜 
    now_time = datetime.today().strftime("%Y%m%d_%H%M%S")[2:]
    # 함수 실행하자 찍힌 초(second)
    in_second = datetime.today().strftime("%S")

    # PATH
    original_path = base_path + F"original/"
    binary_path = base_path + F"binary/"
    original_img = original_path + F"original.{now_time}.jpg"
    binary_img = binary_path + F"binary.{now_time}.jpg"

    # 카메라 설정(해상도 1280x1024)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    while True:    
        ret, frame = cap.read()
        # 카메라가 켜지고 나서 찍힌 초(second)
        now_second = datetime.today().strftime("%S")
        
        # 바이너리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 함수 실행부터 카메라가 켜진 경과 초(second)
        # - 함수 실행해서 바로 찍으면 카메라(Dino)에 조명이 채 들어오기 전에 찍혀서
        #   사진이 어둡다. 조명이 들어올 때 까지 시간을 기다려주기 위해 만듦
        shut_second = abs(int(in_second) - int(now_second))

        # 개발용 노트북 기준으로는 최소 4초 이상은 걸림
        # - 컴퓨터가 바뀌면 아 수치도 바뀔 가능성 있음
        if shut_second >= 4 and shut_second <= 8:
            # 원본 / 바이너리 형식으로 한 장씩 저장하고 나감
            cv2.imwrite(original_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(binary_img, dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
            break

    cap.release()
    cv2.destroyAllWindows()

    # 각각의 사진이(원본, 바이너리) 저장된 폴더 경로
    original_file_list = os.listdir(original_path)
    binary_file_list = os.listdir(binary_path)
    
    # 각각의 경로에서 가장 나중에 찍은(가장 최신의) 사진파일
    original_capture_file = original_path + original_file_list[len(original_file_list)-1]
    binary_capture_file = binary_path + binary_file_list[len(binary_file_list)-1]

    # 을 리턴함
    return original_capture_file, binary_capture_file


# 
# 
# 
def find_circle_area(original_file, binary_file, base_path):
    
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
    
    # PATH
    circle_path = base_path + F"circle/"
    cropped_path = base_path + F"circle_cropped/"
    
    # 카메라에서 가장 나중에 찍었던 사진(오리지널, 바이너리)
    original_img = Image.open(original_file)
    binary_img = Image.open(binary_file)

    # 원형 검출 시 검출해야 할 사이즈
    # - 바이너리 파일에서 원이라고 판단되는 구역을 찾을 때
    #   해당 원 사이즈를 미리 설정
    # - 이걸 안 해놓으면 원이 아닌 것도 원이라고 잡는 경우가 있다
    # - 사진 해상도는 1280x1024 기준이다
    min_size, max_size = 140, 170

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
    visual.save(circle_path + F"{now_time}.jpg")


# 
# 
# 
def rotate_image(base_path):
    
    # PATH
    orig_ng_path = base_path + F"circle_cropped/ng/" 
    rota_ng_path = base_path + F"circle_cropped_rotate/ng/"
    orig_safe_path = base_path + F"circle_cropped/safe/"    
    rota_safe_path = base_path + F"circle_cropped_rotate/safe/"

    # 부품 내 동그란 부분만 크롭된 사진 파일이 있는 곳
    # - 초기에는 Safe와 NG를 수동으로 나눠서 보관해야한다
    ng_img_list = os.listdir(orig_ng_path)
    safe_img_list = os.listdir(orig_safe_path)

    # safe폴더와 ng폴더 안의 사진파일 갯수가 다르므로 반복문을 따로 돌린다
    # - 안에서 하는 일은 똑같다
    # - 원본 / 원본90도회전 / 원본180도회전 / 원본270도회전
    #   좌우반전 / 좌우반전90도회전 / 좌우반전180도회전 / 좌우반전270도회전
    #   시킨 파일을 저장한다
    # - 이 파일로 학습을 시킨다
    for i in ng_img_list:

        i_name = i[:len(i)-4]
        
        img = cv2.imread(orig_ng_path + i, cv2.IMREAD_COLOR) 
        img_flip = cv2.flip(img, 1) # 1은 좌우 반전, 0은 상하 반전입니다.
        img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
        img_180 = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전
        img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 = 시계방향으로 270도 회전
        img_flip_90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
        img_flip_180 = cv2.rotate(img_flip, cv2.ROTATE_180) # 180도 회전
        img_flip_270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 = 시계방향으로 270도 회전

        file_name = F"{rota_ng_path}ng.{i_name}"

        cv2.imwrite(F"{file_name}_orig_0.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_90.jpg", img_90, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_180.jpg", img_180, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_270.jpg", img_270, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_0.jpg", img_flip, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_90.jpg", img_flip_90, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_180.jpg", img_flip_180, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_270.jpg", img_flip_270, [cv2.IMWRITE_JPEG_QUALITY, 100])

    for i in safe_img_list:

        i_name = i[:len(i)-4]
        
        img = cv2.imread(orig_safe_path + i, cv2.IMREAD_COLOR) 
        img_flip = cv2.flip(img, 1) # 1은 좌우 반전, 0은 상하 반전입니다.
        img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
        img_180 = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전
        img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 = 시계방향으로 270도 회전
        img_flip_90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
        img_flip_180 = cv2.rotate(img_flip, cv2.ROTATE_180) # 180도 회전
        img_flip_270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 = 시계방향으로 270도 회전

        file_name = F"{rota_safe_path}safe.{i_name}"

        cv2.imwrite(F"{file_name}_orig_0.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_90.jpg", img_90, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_180.jpg", img_180, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_orig_270.jpg", img_270, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_0.jpg", img_flip, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_90.jpg", img_flip_90, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_180.jpg", img_flip_180, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(F"{file_name}_flip_270.jpg", img_flip_270, [cv2.IMWRITE_JPEG_QUALITY, 100])


# 
# 
# 
def split_train_validation(cropped_path, dataset_path):

    # 일단 폴더 안에 있는 사진파일을 삭제한다(초기화)
    def delete_all_files(file_path):
        if os.path.exists(file_path):
            for file in os.scandir(file_path):
                os.remove(file.path)
            return "Remove All File"
        else:
            return "Directory Not Found"

    # 파일을 복사한다
    def copy_files(file_list, dir, file_names, copy_path):
        for i in file_list:
            if os.path.exists(dir + "/" + file_names[i]):
                shutil.copy2(dir + "/" + file_names[i], copy_path)

    # 겹치치 않는 난수를 넣을 리스트
    # - train리스트와 validation리스트에 6:4 비율로 겹치치 않는 숫자를 넣음
    # - 그 숫자에 있는 파일을 복사
    ng_train_list, ng_difference_list, ng_validation_list = list(), list(), list()
    safe_train_list, safe_difference_list, safe_validation_list = list(), list(), list()

    # 모델 학습용 ng, safe파일 경로
    ng_dir = os.path.join(cropped_path, 'ng')
    safe_dir = os.path.join(cropped_path, 'safe')

    # 해당 경로 안에 있는 학습용 파일들
    ng_fnames = os.listdir(ng_dir)
    safe_fnames = os.listdir(safe_dir)

    # ng_all_files : 폴더 안 파일의 총 갯수
    # ng_train_files : train에 사용할 파일의 총 갯수(60%)
    ng_all_files = len(ng_fnames)
    ng_train_files = round(ng_all_files * 0.6)
    
    # safe_all_files : 폴더 안 파일의 총 갯수
    # safe_train_files : train에 사용할 파일의 총 갯수(60%)
    safe_all_files = len(safe_fnames)
    safe_train_files = round(safe_all_files * 0.6)

    # 폴더 내부 초기화
    delete_all_files(F"{dataset_path}train/ng/")
    delete_all_files(F"{dataset_path}validation/ng/")
    delete_all_files(F"{dataset_path}train/safe/")
    delete_all_files(F"{dataset_path}validation/safe/")

    # 난수의 차집합을 구할 때 쓸 리스트
    for i in range(ng_all_files):
        ng_difference_list.append(i)
    for i in range(safe_all_files):
        safe_difference_list.append(i)

    # 폴더 안 파일의 총 갯수 사이의 난수
    ng_ran_num = random.randint(0,ng_all_files-1)
    safe_ran_num = random.randint(0,safe_all_files-1)

    # train에 사용할 파일의 총 갯수(60%) 만큼 반복
    for i in range(ng_train_files):
        # 중복을 피하여 난수를 발생시켜
        while ng_ran_num in ng_train_list:
            ng_ran_num = random.randint(0,ng_all_files-1)
        # train리스트에 넣는다
        ng_train_list.append(ng_ran_num)

    # train에 사용할 파일의 총 갯수(60%) 만큼 반복
    for i in range(safe_train_files):
        # 중복을 피하여 난수를 발생시켜
        while safe_ran_num in safe_train_list:
            safe_ran_num = random.randint(0,safe_all_files-1)
        # train리스트에 넣는다
        safe_train_list.append(safe_ran_num)

    # 일단 정렬을 하기는 하는데 꼭 필요한가 싶다
    ng_train_list.sort()
    safe_train_list.sort()

    # 차집합을 구하기 위한 로직
    ng_set1, ng_set2 = set(ng_train_list), set(ng_difference_list)
    safe_set1, safe_set2 = set(safe_train_list), set(safe_difference_list)

    # validation리스트 : 전체 - train리스트를 뺀 나머지
    ng_validation_list = list(ng_set2.difference(ng_set1))
    safe_validation_list = list(safe_set2.difference(safe_set1))

    # 각각의 리스트를 해당 경로에 복사
    copy_files(ng_train_list, ng_dir, ng_fnames, F"{dataset_path}train/ng/")
    copy_files(ng_validation_list, ng_dir, ng_fnames, F"{dataset_path}validation/ng/")
    copy_files(safe_train_list, safe_dir, safe_fnames, F"{dataset_path}train/safe/")
    copy_files(safe_validation_list, safe_dir, safe_fnames, F"{dataset_path}validation/safe/")


# 
# 
# 
def create_model(dataset_path, model_path):
    
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'validation')

    # 훈련에 사용되는 NG/SAFE 이미지 경로
    train_ng_dir = os.path.join(train_dir, 'ng')
    train_safe_dir = os.path.join(train_dir, 'safe')

    # 테스트에 사용되는 NG/SAFE 이미지 경로
    validation_ng_dir = os.path.join(validation_dir, 'ng')
    validation_safe_dir = os.path.join(validation_dir, 'safe')

    # 데이터세트 만들기
    # 로더에 대한 몇 가지 매개변수를 정의합니다.
    batch_size = 32
    img_height = 100
    img_width = 100
    epochs_count = 70

    # 파일 이름과 개수
    # os.listdir() 메서드는 경로 내에 있는 파일의 이름을 리스트의 형태로 반환합니다.
    # train_ng_fnames = os.listdir(train_ng_dir)
    # train_safe_fnames = os.listdir(train_safe_dir)
    # print('Total training ng images :', len(os.listdir(train_ng_dir)))
    # print('Total training safe images :', len(os.listdir(train_safe_dir)))
    # print('Total validation ng images :', len(os.listdir(validation_ng_dir)))
    # print('Total validation safe images :', len(os.listdir(validation_safe_dir)))

    # 각 파일의 총 갯수(train, validation)
    train_all_files = len(os.listdir(train_ng_dir)) + len(os.listdir(train_safe_dir))
    validation_all_files = len(os.listdir(validation_ng_dir)) + len(os.listdir(validation_safe_dir))
    
    # # 모델 구성하기
    # # 이제 TensorFlow를 이용해서 합성곱 신경망의 모델을 구성합니다.
    # # summary() 메서드를 이용해서 신경망의 구조를 확인할 수 있습니다.
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    # 모델 컴파일하기
    # 모델 컴파일 단계에서는 compile() 메서드를 이용해서 손실 함수 (loss function)와
    # 옵티마이저 (optimizer)를 지정합니다.
    # 말과 사람 이미지 분류하기 예제에서와 같이 손실 함수로 ‘binary_crossentropy’를 사용했습니다.
    # 출력층의 활성화함수로 ‘sigmoid’를 사용했고, 이는 0과 1 두 가지로 분류되는 ‘binary’ 분류 문제에
    # 적합하기 때문입니다. 또한, 옵티마이저로는 RMSprop을 사용했습니다.
    # RMSprop (Root Mean Square Propagation) Algorithm은 훈련 과정 중에 학습률을 적절하게 변화시킵니다.
    model.compile(optimizer=RMSprop(lr=0.001),
                loss='binary_crossentropy',
                metrics = ['accuracy'])

    # 이미지 데이터 전처리하기
    # 훈련을 진행하기 전, tf.keras.preprocessing.image 모듈의
    # ImageDataGenerator 클래스를 이용해서 데이터 전처리를 진행합니다.
    # 우선 ImageDataGenerator 객체의 rescale 파라미터를 이용해서 모든 데이터를 255로 나누어준 다음,
    # flow_from_directory() 메서드를 이용해서 훈련과 테스트에 사용될 이미지 데이터를 만듭니다.
    # 첫번째 인자로 이미지들이 위치한 경로를 입력하고, batch_size, class_mode를 지정합니다.
    # target_size에 맞춰서 이미지의 크기가 조절됩니다.
    train_datagen = ImageDataGenerator(rescale = 1.0/255.)
    test_datagen  = ImageDataGenerator(rescale = 1.0/255.)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(img_width, img_height))
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=(img_width, img_height))


    # 모델 훈련하기
    # fit() 메서드는 앞에서 구성한 Neural Network 모델을 훈련합니다.
    # 훈련과 테스트를 위한 데이터셋인 train_generator, validation_generator를 입력합니다.
    # epochs는 데이터셋을 한 번 훈련하는 과정을 의미합니다.
    # steps_per_epoch는 한 번의 에포크 (epoch)에서 훈련에 사용할 배치 (batch)의 개수를 지정합니다.
    # - train안에 있는 총 데이터 수
    # validation_steps는 한 번의 에포크가 끝날 때, 테스트에 사용되는 배치 (batch)의 개수를 지정합니다.
    # - validation안에 있는 총 데이터 수
    # → 문제는 매개 변수와 로 나눈 데이터 요소의 총 수와 같아야한다는 사실에서 비롯됩니다.
    #   steps_per_epochvalidation_stepsbatch_size
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=int(train_all_files/batch_size),
                        epochs=epochs_count,
                        validation_steps=int(validation_all_files/batch_size),
                        verbose=2)

    model.save(model_path + 'model6X.h5')

    # 정확도와 손실 확인하기
    # Matplotlib 라이브러리를 이용해서 훈련 과정에서 에포크에 따른 정확도와 손실을 출력합니다.
    # 아래와 같은 이미지가 출력됩니다.
    # 20회 에포크에서 훈련 정확도는 1.0에 근접한 반면, 테스트의 정확도는 100회 훈련이 끝나도
    # 0.7 수준에 머물고 있습니다. 이러한 현상을 과적합 (Overfitting)이라고 합니다.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'go', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def judge_image(model_path):
        
    test_img = "C:/Users/ROG3070/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/cnn_image/mkh_connector/test.jpg"
    model = load_model(model_path + "model.h5")



def create_model2(dataset_path, model_path):

    test_img = "C:/Users/ROG3070/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/cnn_image/mkh_connector/test.jpg"
    image_size = (100, 100)
    batch_size = 32

    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     dataset_path,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=1337,
    #     image_size=image_size,
    #     batch_size=batch_size,
    # )
    # val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     dataset_path,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=1337,
    #     image_size=image_size,
    #     batch_size=batch_size,
    # )

    # from keras import layers

    # data_augmentation = tf.keras.Sequential(
    #     [
    #         layers.RandomFlip("horizontal"),
    #         layers.RandomRotation(0.1),
    #     ]
    # )

    # augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # train_ds = train_ds.prefetch(buffer_size=32)
    # val_ds = val_ds.prefetch(buffer_size=32)

    # def make_model(input_shape, num_classes):
    #     inputs = tf.keras.Input(shape=input_shape)
    #     # Image augmentation block
    #     x = data_augmentation(inputs)

    #     # Entry block
    #     x = layers.Rescaling(1.0 / 255)(x)
    #     x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation("relu")(x)

    #     x = layers.Conv2D(64, 3, padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation("relu")(x)

    #     previous_block_activation = x  # Set aside residual

    #     for size in [128, 256, 512, 728]:
    #         x = layers.Activation("relu")(x)
    #         x = layers.SeparableConv2D(size, 3, padding="same")(x)
    #         x = layers.BatchNormalization()(x)

    #         x = layers.Activation("relu")(x)
    #         x = layers.SeparableConv2D(size, 3, padding="same")(x)
    #         x = layers.BatchNormalization()(x)

    #         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    #         # Project residual
    #         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
    #             previous_block_activation
    #         )
    #         x = layers.add([x, residual])  # Add back residual
    #         previous_block_activation = x  # Set aside next residual

    #     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation("relu")(x)

    #     x = layers.GlobalAveragePooling2D()(x)
    #     if num_classes == 2:
    #         activation = "sigmoid"
    #         units = 1
    #     else:
    #         activation = "softmax"
    #         units = num_classes

    #     x = layers.Dropout(0.5)(x)
    #     outputs = layers.Dense(units, activation=activation)(x)
    #     return tf.keras.Model(inputs, outputs)

    # model = make_model(input_shape=image_size + (3,), num_classes=2)
    # tf.keras.utils.plot_model(model, show_shapes=True)

    # epochs = 50

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint("save_at_50.h5"),
    # ]
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(1e-3),
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"],
    # )
    # model.fit(
    #     train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    # )

    img = tf.keras.preprocessing.image.load_img(
        test_img, target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # model = load_model(model_path + "model.h5")
    model = load_model("save_at_50.h5")
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent ng and %.2f percent safe."
        % (100 * (1 - score), 100 * score)
    )       



if __name__ == "__main__":
    user_name = getpass.getuser()
    model_path = F"C:/Users/{user_name}/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/model/"    
    capture_path = F"C:/Users/{user_name}/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/cnn_image/mkh_connector/1_capture/"
    cropped_path = F"C:/Users/{user_name}/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/cnn_image/mkh_connector/1_capture/circle_cropped_rotate/"
    dataset_path = F"C:/Users/{user_name}/Documents/Develop/220708_Keras_Ibutsu/Data/keras_cnn/cnn_image/mkh_connector/2_dataset/"

    #・사진을 찍는다
    # - 원본사진 / 바이너리사진을 찍어서 저장한다
    # - 바이너리사진에서 원 검출을 한다
    # - (미구현) 검출 결과는 원본사진에 표시한다
    # original_file, binary_file, = cap_origin_binary(capture_path)
    
    #・사진파일에서 원을 검출한다
    # - 바이너리사진파일에서 원을 검출한다
    # - 원을 찾으면 주변을 테두리 침 / 가운데 점 찍음
    # - 테두리 친 곳을 잘라서 그 부분만 저장한다
    # find_circle_area(original_file, binary_file, capture_path)
    
    #・원본파일을 회전시켜서 저장한다
    # - 0도/90도/180도/270도, 좌우반전 후 0도/90도/180도/270도
    # - 학습 시키기 위해 사진 파일을 다양하게 만들었다
    # - 이걸로 모델 학습을 시킬 것이다
    rotate_image(capture_path)

    #・모델학습 전 이미지 분리
    # - 보통 데이터를 분할할 때, Train:Test = 8:2 (or 7:3)으로 나눕니다.
    #   여기서 Train 데이터 중 일부를 Validaion으로 사용하는 것이기 때문에
    #   Train : Validation : Test = 6 : 2 : 2로 사용한다고 볼 수 있습니다.
    #   Train 데이터의 일부를 모델 검증에 사용하기 때문에 그만큼 학습을 시킬 데이터가 줄어들게 됩니다.
    # - 학습 데이터를 희생하면서까지 Validation 데이터를 만들어야 하는 이유
    #   : 범용적으로 사용할 수 있는 모델을 만드는 것이기 때문
    #   예를 들어, Train data의 성능은 좋은 반면, Validation data의 성능이 낫다면
    #   그 모델은 train data에 과적합(;Overfitting)  되었을 가능성이 큽니다.
    #   그래서 과적합을 막기 위해서 Train의 성능을 좀 포기하더라도 Validation의 성능과
    #   비슷하게 맞춰줄 필요가 있습니다. 
    split_train_validation(cropped_path, dataset_path)

    #・모델학습
    create_model(dataset_path, model_path)

    #・모델을 사용해서 예측
    judge_image(model_path)

    #・모델학습
    # create_model2(dataset_path, model_path)
