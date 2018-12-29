# -*- coding: utf-8 -*-
#
# 顔検出プログラム
# 概要：指定されたファオルダ以下に保存されているすべてのJPEGファイルを探し
# 　　　JPEGファイルの中から顔と思われる部分を切り出して出力先フォルダに保存する。
# Usage : python detect.py <画像フォルダ> <出力フォルダ>

import numpy as np
import cv2
import sys
import os
import math

IMAGE_SIZE = 128
INPUT_SIZE = 640

# usage: ln -s /usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades haarcascades'
#xml_dir = os.path.dirname( os.path.abspath( __file__ ) ) + '/haarcascades'
xml_dir = 'C:\\Users\hiro\Anaconda3\pkgs\libopencv-3.4.2-h20b85fd_0\Library\etc\haarcascades'
#xml_dir = '/home/hiro/.pyenv/versions/anaconda3-5.3.0/envs/tensorflow/share/OpenCV/haarcascades'
face_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_eye.xml'))
mouth_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_mouth.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_nose.xml'))

def find_imagefiles_and_detect_face(inpath, outpath):
    # 階層的にフォルダを探索する
    for curDir, dirs, files in os.walk(inpath):
        # フォルダに存在するファイルの中からJPEGファイルだけを抽出する
        print(curDir)
        for a_file in files:
        	if a_file.endswith(".jpg") or a_file.endswith(".JPG"):
                # 見つけたJPEGファイルから顔を検出する
		        detect_face_rotate('%s/%s' % (curDir, a_file), outpath)

def detect_face_rotate(img_file, out_dir):
    filename = os.path.basename(os.path.normpath(img_file))
    (fn, ext) = os.path.splitext(filename)

    print(img_file)
    input_img = cv2.imread(img_file)
    rows, cols, colors = input_img.shape

    if rows > INPUT_SIZE or cols > INPUT_SIZE:
        if rows > cols:
            input_img = cv2.resize(input_img, (int(cols * INPUT_SIZE / rows), INPUT_SIZE))
        else:
            input_img = cv2.resize(input_img, (INPUT_SIZE, int(rows * INPUT_SIZE / cols)))
        rows, cols, colors = input_img.shape

    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.hypot(rows, cols))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray
    results = []
    face_id_seed = 0

    #5度ずつ元画像を回転し、顔の候補を全部取得
    #for deg in range(-50, 51, 5):
    for deg in range(-5, 6, 5):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = face_cascade.detectMultiScale(rotated, 1.02, 5)
        for (x, y, w, h) in faces:
            face_cand = rotated[y:y+h, x:x+w]

            center = (int(x + w * 0.5), int(y + h * 0.5))
            origin = (int(hypot * 0.5), int(hypot * 0.5))
            r_deg = -deg
            center_org = rotate_coord(center, origin, r_deg)

            resized = face_cand
            if w < IMAGE_SIZE:
                resized = cv2.resize(face_cand, (IMAGE_SIZE, IMAGE_SIZE))

            result = {
                    'face_id': 'f%s' % face_id_seed,
                    'img_resized': resized, #顔候補bitmap(小さい場合zoom)
                    'img': face_cand, #顔候補bitmap(元サイズ)
                    'deg': deg, #回転
                    'frame': (x, y, w, h), #回転状態における中心座標+size
                    'center_org': center_org, #角度0時における中心座標
                    }
            results.append(result)
            face_id_seed += 1

    eyes_id_seed = 0
    eyes_faces = []

    for result in results:
        img = np.copy(result["img_resized"])
        fw,fh = img.shape
        eyes = eye_cascade.detectMultiScale(img, 1.02)
        left_eye = right_eye = None #左上/右上にそれぞれ目が１つずつ検出できればOK
        for (x,y,w,h) in eyes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,0),1)

            if not (fw/6 < w and w < fw/2):
                continue
            if not (fh/6 < h and h < fh/2):
                continue
            if not fh * 0.5 - (y + h * 0.5) > 0: #上半分に存在する
                continue
            if fw * 0.5 - (x + w * 0.5) > 0:
                if left_eye:
                    continue
                else:
                    left_eye = (x,y,w,h)
            else:
                if right_eye:
                    continue
                else:
                    right_eye = (x,y,w,h)

        if left_eye and right_eye:
            result['left_eye'] = left_eye
            result['right_eye'] = right_eye
            eyes_faces.append(result)

    #重複検出を除去
    candidates = []
    for i, result in enumerate(eyes_faces):
        result['duplicated'] = False
        for cand in candidates:
            c_x, c_y = cand['center_org']
            _,_,cw,ch = cand['frame']
            r_x, r_y = result['center_org']
            _,_,rw,rh = result['frame']
            if abs(c_x - r_x) < ((cw+rw)*0.5*0.3) and abs(c_y - r_y) < ((ch+rh)*0.5*0.3): #近い場所にある顔候補
                c_diff = eyes_vertical_diff(cand)
                r_diff = eyes_vertical_diff(result)
                if c_diff < r_diff: #より左右の目の水平位置が近いほうが採用
                    result['duplicated'] = True
                else:
                    cand['duplicated'] = True
        candidates.append(result)
    filtered = list(filter(lambda n: n['duplicated'] == False, candidates))

    finals = []
    #候補に対してさらに口検出チェック
    for item in filtered:
        img = np.copy(item["img_resized"])
        fw,fh = img.shape
        mouthes = mouth_cascade.detectMultiScale(img, 1.02) #faceの中心下部付近にあればOK
        mouth_found = False
        for (mx,my,mw,mh) in mouthes:
            cv2.rectangle(img,(mx,my),(mx+mw,my+mh),(128,128,0),2)
            h_diff = fh/2 - (my+mh/2)
            if h_diff < 0:
                mouth_found = True
                break
   
        if mouth_found:
            finals.append(item)

    #最後にカラー画像として切り出し
    res = []
    for item in finals:
        out_file = crop_color_face(item, input_img, out_dir, fn)

def crop_color_face(item, img, out_dir, fn):
    rows, cols, colors = img.shape
    hypot = int(math.hypot(rows, cols))
    frame = np.zeros((hypot, hypot, 3), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = img

    deg = item['deg']
    M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
    rotated = cv2.warpAffine(frame, M, (hypot, hypot))

    x,y,w,h = item['frame']
    face = rotated[y:y+h, x:x+w]
    face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

    out_file = '%s/%s_%s.jpg' % (out_dir, fn, item['face_id'])
    cv2.imwrite(out_file, face)

def eyes_vertical_diff(face):
    _,ly,_,lh = face["left_eye"]
    _,ry,_,rh = face["right_eye"]
    return abs((ly + lh * 0.5) - (ry + rh * 0.5))

def rotate_coord(pos, origin, deg):
    """
    posをdeg度回転させた座標を返す
    pos: 対象となる座標tuple(x,y)
    origin: 原点座標tuple(x,y)
    deg: 回転角度
    @return: 回転後の座標tuple(x,y)
    @see: http://www.geisya.or.jp/~mwm48961/kou2/linear_image3.html
    """
    x, y = pos
    ox, oy = origin
    r = np.radians(deg)
    xd = ((x - ox) * np.cos(r) - (y - oy) * np.sin(r)) + ox
    xy = ((x - ox) * np.sin(r) + (y - oy) * np.cos(r)) + oy
    return (int(xd), int(xy))

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    param = sys.argv
    if len(param) == 3:
        outdir = param[2]
        indir = param[1]
    elif len(param) == 2:
        outdir = 'out'
        indir = param[1]
    else:
        print('Usage : ',param[0],' <input image dir> [<output image dir>]')
        sys.exit()
    
    curdir = os.getcwd()
    if os.path.isdir(indir) == True:
        inpath = os.path.abspath(indir)
    elif os.path.isdir('%s/%s' % (curdir, indir)) == True:
        inpath = os.path.abspath('%s/%s' % (curdir, indir))
    else:
        print('Invalid input image dir!')
        sys.exit()
    
    if os.path.isdir(outdir) == True:
        outpath = os.path.abspath(outdir)
    elif os.path.isdir('%s/%s' % (curdir, outdir)) == True:
        outpath = os.path.abspath('%s/%s' % (curdir, outdir))
    else:
        os.mkdir('%s/%s' % (curdir, outdir))
        outpath = os.path.abspath('%s/%s' % (curdir, outdir))
    
    find_imagefiles_and_detect_face(inpath, outpath)
