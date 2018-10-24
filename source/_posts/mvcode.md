---
title: 电子元器件质量检测代码整理
top: 10
date: 2018-05-03 15:18:29
categories: codeblock
tags:
- MachineVision
- Python
password: 14520
---

电子元器件质量检测模型，主要过程包括：图像预处理、电子元器件检测与识别、图像配准、图像分割、缺陷分离、质量判定。

<h3 id='post_process'>图像预处理</h3>

<h4 id='undistort'>畸变校正</h4>

根据透镜的光学原理，透镜的任意位置上的折射率应该相等，但由于透镜制造精度以及组装工艺的偏差，造成透镜不同位置上的折射率不完全相等，光线在经过透镜的不同区域会产生不同程度的折射，导致成像上出现扭曲变形。这种几何失真就是图像的畸变，其中畸变的程度从图像中心至边缘依次递增,在边缘处反映得尤为明显。

采用`张正友标定算法`进行畸变校正：

    #imageClibra.py
    # -*- coding: utf-8 -*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import numpy as np
    import cv2
    import glob
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:25:5,0:30:5].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('data/*.jpg')#采集到的标定板图像
    count=0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,190,255,cv2.THRESH_BINARY_INV)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(thresh, (5,6),None)
        print ret
        # If found, add object points, image points (after refining them)
        if ret == True:
            count+=1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (5,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.imwrite('clibrate/clib'+str(count)+'.jpg', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    #求取标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    # cv2.destroyAllWindows()
    # mtx为摄像机矩阵、dist为畸变系数、rvecs为旋转量、tvecs为平移量
    print ret,'\n',mtx,'\n',dist,'\n',rvecs,'\n',tvecs

***
<h4 id='disnoise'>图像去噪</h4>

在电子元器件图像采集过程中可能出现的高斯噪声和脉冲噪声都属于加性噪声，可以通过空间滤波器完成图像的复原操作。

<h5 id='addnoise'>添加噪声</h5>

    #getNoise.py
    # -*- coding: utf-8 -*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import cv2
    import numpy as np
    import skimage

    #读取原始图像
    img=cv2.imread('hege/01.jpg',flags=0)

    #添加高斯噪声
    img2=skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)

    #添加脉冲噪声，amount为脉冲噪声的密度
    img3=skimage.util.random_noise(img, mode='salt', seed=None, clip=True,amount=0.005)

    #在添加高斯噪声基础上加入脉冲噪声
    img4=skimage.util.random_noise(img2, mode='salt', seed=None, clip=True,amount=0.005)

    #保存图像
    cv2.imwrite('LenaWgauss.jpg',img2*255)
    cv2.imwrite('LenaWsalt.jpg',img3*255)
    cv2.imwrite('LenaWboth.jpg',img4*255)

添加噪声函数说明：

    skimage.util.random_noise(image, mode=’gaussian’, seed=None, clip=True, **kwargs)

    - image : ndarray.
      Input image data. Will be converted to float.

    - mode : str.
      One of the following strings, selecting the type of noise to add:
        ‘gaussian’ Gaussian-distributed additive noise.
        ‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
        ‘poisson’ Poisson-distributed noise generated from the data.
        ‘salt’ Replaces random pixels with 1.
        ‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
        ‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signedimages.
        ‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.

    - seed : int.
      If provided, this will set the random seed before generating noise,for valid pseudo-random comparisons.

    - clip : bool
      If True (default), the output will be clipped after noise appliedfor modes ‘speckle’, ‘poisson’, and ‘gaussian’. This isneeded to maintain the proper image data range. If False, clippingis not applied, and the output may extend beyond the range [-1, 1].

    - mean : float
      Mean of random distribution. Used in ‘gaussian’ and ‘speckle’.Default : 0.

    - var : float
      Variance of random distribution. Used in ‘gaussian’ and ‘speckle’.Note: variance = (standard deviation) ** 2. Default : 0.01

    - local_vars : ndarray
      Array of positive floats, same shape as image, defining the localvariance at every image point. Used in ‘localvar’.

    - amount : float
      Proportion of image pixels to replace with noise on range [0, 1].Used in ‘salt’, ‘pepper’, and ‘salt & pepper’. Default : 0.05

    - salt_vs_pepper : float
      Proportion of salt vs. pepper noise for ‘s&p’ on range [0, 1].Higher values represent more salt. Default : 0.5 (equal amounts)

    Output
    - out : ndarray
      Output floating-point image data on range [0, 1] or [-1, 1] if theinput image was unsigned or signed, respectively.

***
<h5 id='imgfilter'>图像滤波</h5>

    #均值滤波
    avaImg=cv2.blur(img4, (3,3))
    cv2.imshow('avaImg',avaImg)
    cv2.waitKey()

    #高斯滤波
    gassImg = cv2.GaussianBlur(img4,(3,3),0)
    cv2.imshow('gassImg',gassImg)
    cv2.waitKey()

    #中值滤波
    img4=cv2.imread('LenaWboth.jpg',flags=0)
    medianImg = cv2.medianBlur(img4,3)
    cv2.imshow('medianImg',medianImg)
    cv2.waitKey()
***

<h5 id='adjustAlf'>自适应阿尔法滤波</h5>

修正的阿尔法均值滤波器，选取修正参数d值（d在中取值），在像素点处的的邻域中，去除灰度值最高的0.5d个像素和灰度值最低的0.5d个像素，用剩余的像素点的均值来代替元像素点的灰度值，以此去除图像噪声。

    #adjustAlf.py
    # -*- coding: utf-8 -*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import Image

    #计算峰值信噪比
    def psnr(A, B):
        return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)

    #计算平方均值
    # def mse(A, B):
    #     return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)

    #获取剩余像素点并求均值
    def mean_filter(im,x, y, step,d):
        sum_s= im[(x-int(step/2)):(x+int(step/2)+1),(y-int(step/2)):(y+int(step/2)+1)]
        # print sum_s
        sum_ns=np.sort(sum_s.flatten())
        newSum=sum_ns[int(0.5*d):(len(sum_s)-int(0.5*d))]
        sum_s = np.mean(np.array(newSum))
        return sum_s

    #遍历图像像素点
    def alf_filter(img,size,d):
        im_copy_med=np.zeros(img.shape)
        for i in range(int(size/2),im_copy_med.shape[0]-int(size/2)):
            for j in range(int(size/2),im_copy_med.shape[1]-int(size/2)):
                # print i,j
                im_copy_med[i][j] = mean_filter(img,i, j,size,d)
        return im_copy_med

***

<h3 id='objdetect'>目标检测识别</h3>

基于YOLO网络的检测方法将候选框提取、特征提取、目标分类、目标定位统一于一个神经网络中。神经网络直接从图像中提取候选区域，通过整幅图像特征来预测目标元件的位置和概率，将目标元件的定位问题转化为回归问题，实现端到端(end to end)的检测。目标元件的检测就是对输入的图像，进行候选框提取，判断其中是否包含目标元件，若有给出其位置。

<h4 id='yoloconfig'>yolo配置</h4>

网络结构配置

    #yolo.cfg
    [net]
    batch=64#批量大小
    subdivisions=8#批量细分为8块
    height=416#图片大小
    width=416
    channels=3#通道数
    momentum=0.9#上次梯度更新的权值
    decay=0.0005#权重衰减值
    #图像处理参数，增加样本数
    angle=0#旋转
    saturation = 1.5#透明度
    exposure = 1.5#明暗
    hue=.1#色彩饱和度

    learning_rate=0.0001#初始学习率
    max_batches = 25000#最大迭代次数
    policy=steps#学习率调整策略
    steps=500,2000,5000,12000
    scales=10,.1,.1,.1#在不同Step对应学习率

    [convolutional]#卷积层
    batch_normalize=1#是否进行归一化(BN)
    filters=32#卷积核个数
    size=3#卷积核大小
    stride=1#滑动步长
    pad=1#边缘扩充
    activation=leaky#激活函数

    [maxpool]#最大池化层
    size=2
    stride=2
    .........
    .........
    [route]#passthrough融合其它卷积层特征
    layers=-9#当前层数-9

    [reorg]#特征重组层
    stride=2#相邻随机取一个，26*26*512->13*13*2048

    [route]
    layers=-1,-3#以该层为基准，取向上1层和3层输出，进行向量拼接

    [convolutional]
    batch_normalize=1
    size=3
    stride=1
    pad=1
    filters=1024
    activation=leaky

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=30
    activation=linear

    [region]
    #锚点规格
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=1#类别数
    coords=4#位置框坐标数(x,y,w,h)
    num=5#每个网格预测框数
    softmax=1
    jitter=.2
    rescore=1

    object_scale=5#包含objict的loss系数
    noobject_scale=1#不包含objct的loss系数
    class_scale=1#类别的loss系数
    coord_scale=2#目标位置的loss系数

    absolute=1
    thresh = .6#位置框判断阈值
    random=0#是否使用多尺度训练

训练参数配置

    #yolo.data
    #类别数
    classes= 1
    #训练集文件list
    train  = /home/gz/Desktop/chips_yolo/gen_image_lables/train/train_list.txt
    #验证集list
    valid  = /home/gz/Desktop/chips_yolo/gen_image_lables/train/valid_list.txt
    #类别名称
    names = /home/gz/Desktop/chips_yolo/data/pin.name
    #模型参数路径
    backup = backup

***
<h4 id='dataEnhan'>数据增强</h4>

<h5 id='manulable'>人工标记框</h5>

    #objectpostion.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    #根据人工框选生成原始标签
    import numpy as np
    import cv2

    data_dir='../data/chip/'
    image_name=data_dir+'1_01.jpg'#图片路径
    img = cv2.imread(image_name)

    file_dir='label/'#标签存储路径

    drawing = False #鼠标按下为真
    mode = True #如果为真，画矩形，按m切换为曲线
    ix,iy=-1,-1
    px,py=-1,-1
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,px,py

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy=x,y
        # elif event == cv2.EVENT_MOUSEMOVE:
        #     if drawing == True:
        #         cv2.rectangle(img,(ix,iy),(px,py),(0,0,0),5)#将刚刚拖拽的矩形涂黑
        #         cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),5)
        #         px,py=x,y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # # add rectangle label
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
            cv2.imwrite('chips_label.jpg', img)
            file.write(str(ix)+'\t'+str(iy)+'\t'+str(x)+'\t'+str(y)+'\n')
            px,py=-1,-1
    cv2.namedWindow('image')

    file=open(file_dir+image_name.split('.')[-2].split('/')[-1]+'_origin.txt','w')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') :
            file.close()
            break
        elif k == 27:
            file.close
            break
    cv2.destroyAllWindows()

***
<h5 id='gendata'>生成数据及标签</h5>

模块函数

    #generate.py
    # -*-coding=utf8-*-
    #用于生成图片、矫正图片位置

    import numpy as np
    import os
    from keras.preprocessing.image import transform_matrix_offset_center, array_to_img, img_to_array, load_img,apply_transform
    import scipy as sp
    import scipy.misc
    # import imreg_dft as ird
    import matplotlib.pyplot as plt

    #随机生成变换参数
    def param_random(x,rotation_range,height_shift_range,width_shift_range,shear_range,zoom_range):
        img_row_axis = x.shape[0]-1
        img_col_axis = x.shape[1]-1
        img_channel_axis = x.shape[2]-1
        # use composition of homographies
            # to generate final transform that needs to be applied
        if height_shift_range:
            tx = np.random.uniform(-height_shift_range, height_shift_range) * img_row_axis
        else:
            tx = 0

        if width_shift_range:
            ty = np.random.uniform(-width_shift_range, width_shift_range) * img_col_axis
        else:
            ty = 0
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
        else:
            theta = 0
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx= np.random.uniform(zoom_range[0], zoom_range[1])
            zy=zx
        return tx,ty,theta,shear,zx,zy

    #根据变换参数进行图像变换
    def image_trans(x,tx,ty,theta,shear,zx,zy):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = x.shape[0]-1
        img_col_axis = x.shape[1]-1
        img_channel_axis = x.shape[2]-1

        transform_matrix = None

        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = img_row_axis, img_col_axis
            # print transform_matrix
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode='nearest')

        return x,transform_matrix

    #生成图片
    def gen_image(x,count,result,rotation_range,height_shift_range,width_shift_range,shear_range,zoom_range,flag,save_dir,save_prefix,save_format):
        tx,ty,theta,shear,zx,zy=param_random(x,rotation_range,height_shift_range,width_shift_range,shear_range,zoom_range)
        # print tx,ty,theta,shear,zx,zy
        x,transform_matrix=image_trans(x,tx,ty,theta,shear,zx,zy)
        #生成位置标签
        center_y=eval(result[0]+'+'+result[2])/2
        center_x=eval(result[1]+'+'+result[3])/2
        cord=np.array([center_x,center_y]).T
        # print transform_matrix
        matrix=transform_matrix[:2,2]
        rotate=np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
        new_matrix=np.dot(np.array([[1/zx, 0],
                                    [0, 1/zy]]),rotate)
        label_center=np.dot(new_matrix,cord-matrix)
        top_left=np.dot(new_matrix,np.array([eval(result[1]),eval(result[0])]).T-matrix)
        top_right=np.dot(new_matrix,np.array([eval(result[1]),eval(result[2])]).T-matrix)
        bottom_left=np.dot(new_matrix,np.array([eval(result[3]),eval(result[0])]).T-matrix)
        bottom_right=np.dot(new_matrix,np.array([eval(result[3]),eval(result[2])]).T-matrix)
        ##########
        img = array_to_img(x)

        save_to_dir=save_dir+'/gen_'+flag
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        fname = '{prefix}_{count}.{format}'.format(prefix=save_prefix,count=count,format=save_format)
        img.save(os.path.join(save_to_dir, fname))
        return tx,ty,theta,shear,zx,zy,label_center,top_left,top_right,bottom_left,bottom_right

    #校正位置标签
    def gen_label(origin_label,label_center,top_left,top_right,bottom_left,bottom_right,size):
        rect_axis=np.array([top_left,top_right,bottom_left,bottom_right])
        [bottom,right]=np.max(rect_axis,0)
        [top,left]=np.min(rect_axis,0)

        if bottom>size[0]:
            bottom=size[0]
        if right>size[1]:
            right=size[1]
        if top<0:
            top=0
        if left<0:
            left=0
        return np.ceil(top),np.ceil(left),np.ceil(bottom),np.ceil(right)

根据图像和原始标签生成数据：

    #gen_data.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    #生成标签数据
    #输入：原图label
    #输出：图片及标签

    import numpy as np
    import os
    import scipy.misc
    import matplotlib.pyplot as plt
    from generate import *

    rotation_range=90 #旋转角度
    height_shift_range=0.1#纵向偏移
    width_shift_range=0.1#横向偏移
    shear_range=0#剪切变换
    zoom_range=[1.5,3]#缩放
    num=20#生成图片数目
    flag=str(num)#文件夹标志
    save_dir='../data'#图片存储路径
    label_dir='./label/'#标签存储路径
    save_prefix='pin'
    save_format='jpg'
    orig_label=label_dir+'1_01_origin.txt'
    img = load_img('../data/chip/1_01.jpg')  # this is a PIL image

    # 存储图片标签文档
    file1=open(label_dir+'image_label.txt','w')
    # 存储图片变换参数
    trans_label=open(label_dir+'trans_label.txt','w')
    # 读取原始标签
    file2=open(orig_label,'r')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    origin_label=file2.readline().split()
    file2.close()

    x = img_to_array(img)
    size=x.shape

    for i in range(num):
        tx,ty,theta,shear,zx,zy,label_center,top_left,top_right,bottom_left,bottom_right=gen_image(x,i,origin_label,rotation_range,height_shift_range,width_shift_range,shear_range,zoom_range,flag,save_dir,save_prefix,save_format)

        trans_label.write(str(i)+'\t'+str(tx)+'\t'+str(ty)+'\t'+str(theta)+'\t'+str(zx)+'\t'+str(zy)+'\n')
        # print tx,ty,theta,shear,zx,zy
        h1,w1,h2,w2=gen_label(origin_label,label_center,top_left,top_right,bottom_left,bottom_right, size)
        if (i+1)%10==0:
            print 'generate.....'+str(i+1)
        file1.write(str(i)+'\t'+str(int(w1))+'\t'+str(int(h1))+'\t'+str(int(w2))+'\t'+str(int(h2))+'\n')
    file1.close()

***
<h5 id='genlist'>生成训练list</h5>

生成图片路径list

    #train/getimagelist.py
    #Generate image Path
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import os

    #图片路径
    base_dir='/Users/gz/Workspace/chips_yolo/gen_image_lables/data/gen_2000'
    #list存储路径
    train='/Users/gz/Workspace/chips_yolo/gen_image_lables/train'

    files=os.listdir(base_dir)

    imagepath=open(train+'/imagepath.txt','w');
    for file in files:
        if file!='.DS_Store':
            genpath=os.path.join(base_dir,file);
            imagepath.write(genpath+'\n');
    print 'imagepath.txt is complete'
    imagepath.close()

生成训练和验证list

    #train/train_list.py
    #Generate train_list and label_list
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import random
    import numpy as np
    import os
    import re

    path='/Users/gz/Workspace/chips_yolo/gen_image_lables/'
    #图片list路径
    infile=open(path+'train/imagepath.txt','r')
    #读取标签数据文档
    label_file = open(path+'gendata/label/image_label.txt')
    #单独标签文档存储路径
    labelTxtPath='../data/gen_2000'

    #位置标签归一化
    def convert_labels(line):
        data=line.strip().split()
        # print line
        img_num=data[0]
        dw=1./1280
        dh=1./960
        x=(float(data[1])+float(data[3]))/2*dw
        y=(float(data[2])+float(data[4]))/2*dh
        w=(float(data[3])-float(data[1]))*dw
        h=(float(data[4])-float(data[2]))*dh
        # out_file.write('')
        return '0'+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n'

    #生成标签文档
    def gen_labelfile(label_path,labelTxtPath):
        if(not os.path.exists(labelTxtPath)):
            os.makedirs(labelTxtPath)
        for label in label_path:
            singlelabel=open('labelTxtPath/chips_'+label.split()[0]+'.txt','w')
            singlelabel.write(convert_labels(label))
            singlelabel.close()

    #Training Path
    nums=0
    lines=[]
    labels=[]
    #save path in Array
    for line in infile:
        lines.append(line)
        nums+=1

    #random index，生成随机数
    train_num=np.floor(nums*0.8)
    vaild_num=nums-train_num
    index=[i for i in range(nums)]
    random.shuffle(index)

    outfile=open(path+'train/train_list.txt','w')
    validfile=open(path+'train/valid_list.txt','w')

    for n in index:
        if(n<=train_num):
            outfile.write(lines[n])
        else:
            validfile.write(lines[n])

    infile.close()
    outfile.close()
    validfile.close()

    #Generate label
    gen_labelfile(label_file,labelTxtPath)

***
<h5 id='train'>模型训练</h5>

以开源的深度神经网络框架[Darknet](https://pjreddie.com/darknet/)为基础，搭建网络模型，并进行训练。

    ./darknet detector train cfg/yolo.data cfg/yolo.cfg darknet53.conv.74

对单张图片进行检测

    ./darknet detector test cfg/yolo.data cfg/yolo.cfg yolo.weights data/dog.jpg -thresh 0

在验证集上进行检测

    ./darknet detector recall cfg/yolo.data cfg/yolo.cfg yolo.weights

***

<h3 id='registration'>图像配准</h3>

傅里叶配准算法的核心是相位相关法，两幅具有平移量的图像经过傅里叶变换后，它们在频域上表现为相位不同；具有旋转量的图像在经过傅里叶变换后，在频域上表现为相同的旋转量；具有尺度缩放的图像，极坐标下经傅里叶变换后，映射到对数坐标系下可以转化为平移量进行处理。

<h4 id='resize'>图像变换到相同尺寸</h4>

首先对图像进行畸变校正，然后将其变换到相同尺寸

    #align_test.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import os
    import scipy as sp
    import scipy.misc
    import time
    import Image
    from matplotlib import pyplot as plt
    import numpy as np
    import imreg_dft as ird
    import sys
    sys.path.append("binary/")
    from iter_best import *

    #摄像机内参
    mtx=np.array([[5526.15,0,558.216],[0,5.52973614e+03,4.67279273e+02],[0,0,1]])
    dist=np.array([[-3.87550438e-01,1.67434651e+00,-7.78018545e-04,6.76212640e-03,-7.95626592e-02]])

    #畸变校正函数
    def clibrate(im):
        img=cv2.imread(im)
        h, w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(dst)
        # plt.show()
        return dst

    #图片resize,rect1和rect2分别为图像上目标元件的位置信息
    def ImageResize(img1,img2,rect1=None,rect2=None):

    im1=Image.fromarray(clibrate(img1))
    im2=Image.fromarray(clibrate(img2))
    # 读取图片
    # im1=Image.open(img1)
    # im2=Image.open(img2)
    # 剪切图片
    # print rect1,rect2
    if not (rect1 or rect2):
        region1=im1
        region2=im2
        size1=im1.size
        size2=im2.size
        # print size1,size2
    else:
        region1=im1.crop(rect1)
        region2=im2.crop(rect2)

        # 图片填充至一样大小
        size1=region1.size
        size2=region2.size

    #转化为数组
    im1_array=np.array(region1)
    im2_array=np.array(region2)

    # 取最大背景尺寸，并填充
    size=max(size1[1],size2[1]),max(size1[0],size2[0])
    new_array1=np.ones(size)*im1_array[0][0][0]
    new_array2=np.ones(size)*im2_array[0][0][0]

    background1=Image.fromarray(new_array1)
    background2=Image.fromarray(new_array2)

    # print size1[0]
    # 图像粘贴
    background1.paste(region1,(0,0))
    background2.paste(region2,(0,0))

    # 保存图像
    save_path='Data/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    path1=save_path+img1.split('.')[0]+'1_resize.jpg'
    path2=save_path+img2.split('.')[0]+'2_resize.jpg'

    if background1.mode!='RGB':
        background1=background1.convert('RGB')
        background1.save(path1)
    if background2.mode!='RGB':
        background2=background2.convert('RGB')
        background2.save(path2)
    return path1,path2

<h4 id='regist'>图像配准</h4>

运用imreg_dft模块进行图像配准。

    #align_test.py
    def convertTonum(arrAy):
        #字符转数字函数
        newArray=[]
        for num in arrAy:
            newArray.append(int(num))
        return newArray

    def align(image1,image2,truth,template,position,show,num,aligned):
        #image1,image2:图片路径
        #truth:真实变换参数文档路径(用于计算配准误差)，txt文档
        #template:模板图像中目标元件位置标签，txt文档
        #position:检测出的目标位置信息，txt文档
        #show:是否显示图片，true/false
        #num:图片编号，string
        #aligned:配准后图像的存储路径

        if (template!=None)&(position!=None):
            rect1=convertTonum(template.split())
            rect2=convertTonum(position.split()[1:])
            # print image1
            path1,path2=ImageResize(image1, image2, rect1, rect2)

            im0=binaryIm(image1)
            im1=binaryIm(image2)
        # else:
        # # im0=imClosing(image1, 3, 20)
        # # im1=imClosing(image2, 3, 20)
        # # print image1
        # # print image2
        # im0=binaryIm(image1)
        # im1=binaryIm(image2)

        #统计配准时间
        begin=time.time()
        result = ird.similarity(image1, image2, numiter=3)
        end=time.time()
        # print 'simTime:'+str(end-begin)

        #变换参数，平移量tx、ty，旋转量theta(弧度)，缩放系数scale
        tx=result.get('tvec','not found')[0]
        ty=result.get('tvec','not found')[1]
        theta=result.get('angle','not found')*np.pi/180
        scale=result.get('scale','not found')

        #校正变换平移量
        ttx=np.cos(theta)*tx+np.sin(theta)*ty
        tty=-np.sin(theta)*tx+np.cos(theta)*ty
        assert "timg" in result
        # Maybe we don't want to show plots all the time
        if show:
            import matplotlib.pyplot as plt
            ird.imshow(image1, image2, result['timg'])
            plt.show()

        # print "aliging"+str(num)
        align_result=aligned
        if not os.path.exists(align_result):
            os.mkdir(align_result)

        #保存配准后图片
        align_save=Image.fromarray(result['timg'])
        # temp_save=Image.fromarray(im0)
        if align_save.mode!='RGB':
            align_save=align_save.convert('RGB')
            align_save.save(align_result+str(num)+'_align.jpg')
        # if temp_save.mode!='RGB':
        # temp_save=temp_save.convert('RGB')
        # temp_save.save(align_result+num+'_temp.jpg')

        retIm=[]
        ##对配准图像进行二值化处理
        #retIm=binaryIm(align_result+str(num)+'_align.jpg',True)

        #计算配准误差
        if truth:
            label=truth.split()
            return ttx,label[1],abs(ttx-eval(label[1])),tty,label[2],abs(tty-eval(label[2])),theta,label[3],abs(theta-eval(label[3])),scale,label[4],abs(scale-eval(label[4])),end-begin
        # x位移、y位移、旋转角、尺度、计算时间
        return ttx,tty,theta,scale,end-begin,retIm

***

<h3 id='segment'>图像分割</h3>

主要考虑采用全局阈值分割方法进行元件的形状轮廓的提取。

<h4 id='dtop'>双峰阈值分割</h4>

双峰阈值分割方法是根据图像的灰度直方图上出现明显双峰进行图像阈值选取，阈值取两峰之间的波谷。

    #binary/doubleTop.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    # -*-coding=utf8-*-
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    def dtop(im,minPix=0,maxPix=255):
    #minPix、maxPix需要检测的灰度值区间
    #返回值：分割阈值
        Iter=0
        length=maxPix-minPix
        hist=cv2.calcHist(im, [0], None,[length], [minPix,maxPix])
        # plt.plot(hist)
        # plt.show()
        while isDiodal(hist)==False:
            # plt.plot(hist48.jpg
            hist[0]=(hist[0]*2+hist[1])/3.0
            for x in range(1,length-1):
                hist[x]=(hist[x-1]+hist[x]+hist[x+1])/3.0
            hist[length-1]=(hist[length-1]*2+hist[length-2])/3.0
            Iter+=1
            if(Iter>1000):
                return False
        top,trough=isDiodal(hist)
        for thresh in trough:
            if thresh<top[1] and thresh>top[0]:
                return thresh+minPix

    #判断直方图是否为双峰
    def isDiodal(hist):
        count=0
        top=[]
        trough=[]
        for x in range(1,len(hist)-1):
            if(hist[x-1]<hist[x]) and (hist[x+1]<hist[x]):
                count+=1
                top.append(x)
                # print top
                if count>2:
                    return False
            if(hist[x-1]>hist[x]) and (hist[x+1]>hist[x]):
                trough.append(x)
        if(count==2):
            return top,trough
        else:
            return False

    ##单张图片测试
    # im=cv2.imread("../../data/detect/hege2/48.jpg")
    # threshold=dtop(im,0,255)
    # ret,thresh=cv2.threshold(im, threshold,255,0)
    # plt.imshow(thresh)
    # plt.show()

***
<h4 id='iter'>迭代阈值分割</h4>

给定一个初始阈值，通过该阈值将图像分为前景和背景，分别计算前景和背景的灰度的平均灰度值，将二者的平均值作为新的分割阈值，直至连续两次的分割阈值相等，则该阈值为最佳阈值。

    #binary/iter_best.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    #图像二值化函数
    def imClosing(im,kernel_size,threshold):
        blur=cv2.medianBlur(im,3)#中值滤波
        ret,thresh = cv2.threshold(blur,threshold,255,0)#图像二值化
        if kernel_size==0:
            return thresh
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        #图像闭运算
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return closing

    #阈值迭代
    def iter_best(im):
        # print np.max(im),np.min(im)
        t0=(int(np.max(im))+int(np.min(im)))/2
        # print t0
        # num=1
        while t0!=divIm(im, t0):
            # print num
            # num+=1
            t0=divIm(im, t0)
        # print t0
        return t0

    #前景和背景分割
    def divIm(im,t):
        front=im[im>t]
        back=im[im<=t]

        ava_f=int(np.sum(front)/len(front))
        ava_b=int(np.sum(back)/len(back))

        return (ava_f+ava_b)/2

    #用最佳迭代阈值进行二值化
    def binaryIm(impath,template=False):
        im=cv2.imread(impath,0)
        thresh=iter_best(im)
        # print thresh
        if template:
            # print 'itertime:'+str(time.time()-begin)
            # print thresh
            im0=imClosing(im,0,thresh)
            return im0
        else:
        #在最佳迭代阈值的邻域内，进行多次二值化，得到多幅二值化图像
            im_array=[]
            for i in range(-2,3):
                im_array.append(imClosing(im,0,thresh+i))
            return im_array

***

<h4 id='ostu'>大津算法</h4>

OSTU算法是日本学者OSTU于1979年提出的，是一种自适应阈值的全局阈值分割算法，通过穷举的方法找到一个阈值，使得前景和背景的类间方差最大。

以下代码是经遗传算法优化后的大津算法，代码来源：[`A program for binarizing images based on genetic algorithm`](https://github.com/rafacheng/GeneticAlgo)

    #Ostu_gen.py
    # -*-coding=utf8-*-
    import cv2
    import operator
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import misc

    # calculate fitness
    def calOstu(th, hist):
        omega1 = float(sum(hist[0:th])) / sum(hist)
        omega2 = 1 - omega1
        ip = 0
        for i in xrange(th):
            ip += i * hist[i]
        miu1 = float(ip) / sum(hist)
        ip = 0
        for i in xrange(th, 255):
            ip += i * hist[i]
        miu2 = float(ip) / sum(hist)
        miu = miu1 + miu2
        g = omega1 * (miu1 - miu) ** 2 + omega2 * ((miu2 - miu) ** 2)
        return g

    # calculate cumulative probability
    def calCumPro(totalg, chromosomes):
        newgener = []
        # calculate probability
        for chrom in chromosomes:
            newgener.append([chrom[0], chrom[1] / totalg])
        # calculate  cumulative probability
        for i in xrange(1, len(chromosomes)):
            newgener[i][1] += newgener[i-1][1]
        return newgener

    # choose chromosomes to evolve
    def chooseChrom(chromosomes):
        _sum = 0
        for i in xrange(len(chromosomes)):
            _sum += chromosomes[i][1]
        chromosomes = calCumPro(_sum, chromosomes)
        ps = []
        # get picking-up probability randomly
        for i in xrange(len(chromosomes)):
            ps.append(random.random())
        newgener = []
        # pick up chromosomes
        for p in ps:
            for chrom in chromosomes:
                if p <= chrom[1]:
                    newgener.append(chrom)
                    break
                else:
                    continue
            continue
        return newgener

    # exchange chromosomes
    def exchange(exRate, length, chromosomes):
        exNum = int(exRate * len(chromosomes))
        # make sure even number of chromosomes
        if exNum % 2 == 1:
            exNum -= 1
        # pick up random exNum chromosomes to exchange
        ixes = random.sample(xrange(len(chromosomes)), exNum)
        newgener = []
        # copy other chromosomes
        for i in xrange(len(chromosomes)):
            if i not in ixes:
                newgener.append(chromosomes[i])
        # do exchange
        for i in range(0, len(ixes), 2):
            seg_h1 = chromosomes[ixes[i]][0] >> length << length
            seg_t1 = chromosomes[ixes[i]][0] - seg_h1
            seg_h2 = chromosomes[ixes[i+1]][0] >> length << length
            seg_t2 = chromosomes[ixes[i+1]][0] - seg_h2
            newgener.append([seg_h1 + seg_t2, 0])
            newgener.append([seg_h2 + seg_t1, 0])
        return newgener

    # varying chromosomes
    def vary(varyRate, chromosomes):
        vrNum = int(varyRate * len(chromosomes))
        ixes = random.sample(xrange(len(chromosomes)), vrNum)
        for i in xrange(len(chromosomes)):
            if i in ixes:
                randint = random.randint(0, 31)
                if chromosomes[i][0] > randint:
                    chromosomes[i][0] -= randint
                else:
                    chromosomes[i][0] = randint - chromosomes[i][0]
    # old algorithm, works inefficiently
    #    chromosomes[i][0] = (~(chrom - (chrom >> bits << bits)) \
    #            & (2 ** bits - 1)) + (chrom >> bits << bits)
        return chromosomes

    # binarize image
    def binarize(path, th, img):
        binarized_img = [[255 if x > th else 0 for x in r] for r in img]
        # bin_path = path.split('.')[1] + '_bin.jpg'
        # misc.imsave(bin_path, binarized_img)
        # img = cv2.imread(bin_path, 0)
        return binarized_img

    # calculate best threshold by a naive way
    def naiveFindBestThreshold(path):
        th = 0
        g = 0
        thtemp = 0
        gtemp = 0
        img = cv2.imread(path, 0)
        hist = np.bincount(img.ravel(),minlength=256)
        for thtemp in xrange(256):
            gtemp = calOstu(thtemp, hist)
            if gtemp > g:
                th = thtemp
                g = gtemp
        bin_path = binarize(path, th,img)
        return bin_path, th

    def main(path, cnt, population, cross_ratio, vary_ratio):
        img = cv2.imread(path, 0)
        hist = np.bincount(img.ravel(),minlength=256)
        generation = 1
        s = random.sample(xrange(256), population)
        chromosomes = [[x, 0] for x in s]
        _sum = 0
        length = 4
        lstbest = 0
        count = 0
        exRate = cross_ratio
        varyRate = vary_ratio
        degen = True
        while True:
            # print "generation %d:" % generation
            for i in xrange(len(chromosomes)):
                # show chromosome status
                # print '{0:08b}'.format(chromosomes[i][0])
                chromosomes[i][1] = calOstu(chromosomes[i][0], hist)
                if chromosomes[i][1] >= lstbest:
                    # A better chromosome appears
                    if chromosomes[i][1] > lstbest:
                        count = 0
                    # not worse
                    lstbest = chromosomes[i][1]
                    th = chromosomes[i][0]
                    degen = False
            # if worse, use last generation
            if degen == True:
                for i in xrange(len(chromosomes)):
                    chromosomes[i][0], chromosomes[i][1] = lstgen[i][0], lstgen[i][1]
            else:
                # default set degeneration to True
                degen = True
            # best chromosome stay fixed for 1000 generations, terminate loop
            if count == cnt:
                break
            # calculate sum of between-class variance
            count += 1
            # backup this generation
            lstgen = []
            for chromo in chromosomes:
                lstgen.append([chromo[0], chromo[1]])
            # choose chromosomes to exchange
            chromosomes = chooseChrom(chromosomes)
            # do exchange
            chromosomes = exchange(exRate, length, chromosomes)
            # do vary
            chromosomes = vary(varyRate, chromosomes)
            generation += 1
        # print "best threshold: " + str(th)
        # bin_path = binarize(path, th, img)
        return binarize(path, th, img)

***
<h3 id='diffArea'>元件差异性检测</h3>

经图像配准后，待测元件和标准元件在各自图像上处于同一位置，通过图像处理技术，对比匹配后的图像，分离出电子元器件上的缺陷部分，根据缺陷部分的大小来判断电子元器件是否合格。需要上述函数[align_test.py](#regist)

<h4 id='filled'>孔洞填充</h4>

    #diffDetect.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import os
    import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    #以下为自定义函数
    from align_test import *

    #根据轮廓面积，对轮廓进行排序
    def sortCnt(contours):
        maxarea=0
        length=len(contours)
        for j in range(length-1):
            for i in range(length-1-j):
                area1=cv2.contourArea(contours[i])
                area2=cv2.contourArea(contours[i+1])
                if(area1<area2):
                    temp=contours[i]
                    contours[i]=contours[i+1]
                    contours[i+1]=temp
        return contours

    #对元器件进行轮廓检测
    def drawCnt(im,show=False):
        image, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours)<2:
            # cv2.drawContours(im, contours,0, (0,0,0), cv2.FILLED)
            if show:
                plt.imshow(im,cmap='gray')
                plt.show()
            return im
        newCnt=sortCnt(contours)
        #选出除图形边框外的最大轮廓，对电子元器件的内孔洞进行填充
        cv2.drawContours(im, newCnt,1, (0,0,0), cv2.FILLED)
        if show:
            plt.imshow(im,cmap='gray')
            plt.show()
        return im

***
<h4 id='detect'>缺陷面积检测</h4>

    #diffDetect.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    def diffAreas(image1,image2,num=0,aligned='Image/',show=False):
    #image1、image2:二值化后的图像，array
    #aligned:配准后图像存储路径
    #show:是否显示过程,true/false
        import time
        begin1=time.time()

        ttx,tty,theta,scale,t,im_align=align(image1, image2, None, None, None, show, num,aligned)
        # print ttx,tty,theta
        end1=time.time()-begin1
        print 'aligntime:'+str(end1)
        img1=drawCnt(image1,show)
        img2=drawCnt(im_align,show)

        #差异性检测
        diff = np.array(abs(img1-img2))
        # plt.imshow(diff)
        # plt.show()
        cv2.imwrite('Image/diff.jpg', diff)
        diff_rgb=cv2.imread('Image/diff.jpg')
        # diff_g=cv2.imread('Image/diff.jpg', 0)
        image, contours, hierarchy = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(diff, contours, -1, (0,0,0),cv2.FILLED)

        area=[]
        area_total=0
        new_contour=[]
        cnt_pos=[]

        #对差异部分进行筛选，过滤掉小面积缺陷
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if(w!=1 and w!=0) or (h!=1 and w!=0):
                single_area=cv2.contourArea(contour)
                if(single_area>30):
                    area.append(single_area)
                    area_total=area_total+single_area
                    new_contour.append(contour)
                    cnt_pos.append((x,y,w,h))
        print 'detect:'+str(time.time()-begin1)
        if show:
            # plt.imshow(image,cmap='gray')
            # plt.show()
            cv2.drawContours(diff_rgb, new_contour, -1, (0,255,0), cv2.FILLED)
            plt.imshow(diff_rgb)
            plt.text(300,0,"Areas="+str(area_total),size=15,color='k')
            if(area_total<800):
                plt.text(0,0,"OK",size=20,color='g')
            else:
                plt.text(0,0,"NOT OK",size=20,color='r')
            plt.show()
        return area_total

***

<h4 id='threshold'>阈值确定</h4>

假设合格产品的缺陷面积服从正态分布，并以此提出了基于小概率事件原理的阈值确定方法。运用Shapiro-Wilks方法进行实验验证，Shapiro-Wilks检验是用于验证一个随机样本数据是否来自正态分布。

    #k_test.py
    # -*-coding=utf8-*-
    """
    Created on Tue Mar 15 2018
    @author: Gaozong
    """
    import numpy as np
    from scipy.stats import kstest,shapiro,anderson
    #x为合格样本产生的缺陷面积
    x=[172,198.5,224,355.5,312,377.5,85.5,65.5,74,252,70.5,166,211,161.5,175.5,431.5,288,281,167,195,263.5,330,143,115,43.5,120,224,210.5,253.5,121]

    #排序函数
    def sortCnt(contours):
        maxarea=0
        length=len(contours)
        for j in range(length-1):
            for i in range(length-1-j):
                if(contours[i]>contours[i+1]):
                    temp=contours[i]
                    contours[i]=contours[i+1]
                    contours[i+1]=temp
        print length
        return contours
    #平均值、标准差、阈值、shapiro检验结果
    print np.mean(x)
    print np.std(x)
    print 2*np.std(x)+np.mean(x)
    print shapiro(sortCnt(x))

***