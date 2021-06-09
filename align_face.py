import os
import cv2
import numpy as np
import time


def diff(prev_frame, frame):
    res = frame.astype(np.int16) - prev_frame.astype(np.int16)
    mean = abs(res).mean()
    res = res * 4 + 128
    np.where(res < 0, 0, res)
    np.where(res > 255, 255, res)
    return res.astype(np.uint8), mean


def face_localtion_match(frame, x1, y1, x2, y2):
    global center, size, prev_face, prev_template
    if prev_template is not None:
        # center
        match_result = cv2.matchTemplate(frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])],
                                         prev_template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        tl = max_loc
        center = (tl[0] + size[0] // 8 + max(0, x1), tl[1] + size[1] // 8 + max(0, y1))  # 匹配到的左上角+模板半长宽+目标左上角坐标
        # size
        face_diff_dict = {}
        for i in range(-10, 11, 2):
            sizex = size[0] + i
            sizey = int(sizex * size[1] / size[0])
            xx1 = center[0] - sizex // 2
            xx2 = center[0] + sizex // 2
            yy1 = center[1] - sizey // 2
            yy2 = center[1] + sizey // 2
            # crop
            res_img = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]
            res_img = cv2.resize(res_img, (prev_face.shape[1], prev_face.shape[0]))
            # diff
            res_diff, mean = diff(prev_face, res_img)
            face_diff_dict[mean] = [res_img, res_diff, (sizex, sizey)]
        minres = min(face_diff_dict)
        size = face_diff_dict[minres][2]
        # cv2.imshow("diff", face_diff_dict[minres][1])
        # cv2.waitKey(100)
    else:
        center = ((x2 + x1) // 2, (y2 + y1) // 2)  # read the face position of the first frame
        size = ((x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2)
    xx1 = center[0] - size[0] // 8
    xx2 = center[0] + size[0] // 8
    yy1 = center[1] - size[1] // 8
    yy2 = center[1] + size[1] // 8
    prev_template = frame[yy1:yy2, xx1:xx2]
    xx1 = center[0] - size[0] // 2
    xx2 = center[0] + size[0] // 2
    yy1 = center[1] - size[1] // 2
    yy2 = center[1] + size[1] // 2
    prev_face = frame[max(0, yy1):min(yy2, frame.shape[0]), max(0, xx1):min(xx2, frame.shape[1])]
    return xx1, yy1, xx2, yy2


def face_crop_x16(face_loc, frame):
    h = frame.shape[0]
    w = frame.shape[1]
    x1, y1, x2, y2 = face_loc[-4:]
    # expand
    f_h = y2 - y1
    f_w = x2 - x1
    x1 = x1 - int(0.3 * f_w)
    x2 = x2 + int(0.3 * f_w)
    y1 = y1 - int(0.3 * f_h)
    y2 = y2 + int(0.3 * f_h)
    res_img = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    # crop
    if x1 < x2 and y1 < y2:
        frame = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
        res_img[max(0, y1) - y1:min(y2, h) - y1, max(0, x1) - x1:min(x2, w) - x1] = frame
    else:
        res_img = frame
        print(x1, y1, x2, y2)
    return res_img


center = size = prev_face = None
save_dir = '/home/yaowen/Documents/1Database/PAD-datasets/ReplayAttack/Aligned_faces_x1.6'
jpg_dir = '/home/yaowen/Documents/1Database/PAD-datasets/ReplayAttack/ReplayAttack-Facex2'
location_path = jpg_dir.split('/')[-1] + 'align_face_location.txt'
if os.path.exists(location_path):
    txt = open(location_path, 'r').readlines()
    last_dir = last_depth_dir = ''
    for l in txt:
        l = l.split()
        jpg_path = os.path.join(jpg_dir, l[0])
        jpg_save = os.path.join(save_dir, l[0])
        if os.path.exists(jpg_path + '/' + l[1]) and not os.path.exists(jpg_save + '/' + l[-1]):
            img = cv2.imread(jpg_path + '/' + l[1])
            face = face_crop_x16([int(x) for x in l[2:6]], img)
            if not os.path.exists(jpg_save):
                os.makedirs(jpg_save)
            cv2.imwrite(jpg_save + '/' + l[-1], face)
            if last_dir != jpg_path:
                print('%s align face' % jpg_path)
        else:
            if last_dir != jpg_path:
                print('%s skip face' % jpg_path)
        last_dir = jpg_path
        # # if you've already estimated the face depth from frame, do the following
        # depth_path = os.path.join('/home/yaowen/Documents/1Database/PAD-datasets/CASIA-FASD/depth_map_Facex1.6', l[0])
        # depth_save = os.path.join('/home/yaowen/Documents/1Database/PAD-datasets/CASIA-FASD/Aligned_depthmap_x2', l[0])
        # if os.path.exists(depth_path + '/' + l[1]) and not os.path.exists(depth_save + '/' + l[-1]):
        #     # 有图片却没结果图
        #     img = cv2.imread(depth_path + '/' + l[1])
        #     face = face_crop_x16([int(x) for x in l[2:6]], img)
        #     if not os.path.exists(depth_save):
        #         os.makedirs(depth_save)
        #     cv2.imwrite(depth_save + '/' + l[-1], face)
        #     if last_depth_dir != jpg_path:
        #         print('%s align map' % jpg_path)
        # else:
        #     if last_depth_dir != jpg_path:
        #         print('%s skip map' % jpg_path)
        # last_depth_dir = jpg_path
else:
    txt = open(location_path, 'w')
    t = time.time()
    for path, dirs, files in os.walk(jpg_dir, followlinks=True):
        if len(dirs) == 0:
            print(path, end=' ')
            prev_template = None
            n = 0
            for i in range(len(files)):
                name = '%d.jpg' % i
                jpg_path = os.path.join(path, name)
                frame = cv2.imread(jpg_path)
                if frame is None:
                    print(jpg_path, 'None')
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if n == 0:
                    h, w = frame.shape[:2]
                    xx1, yy1, xx2, yy2 = face_localtion_match(
                        np.copy(frame), int(0.25 * w), int(0.25 * h), int(0.75 * w), int(0.75 * h)
                    )  # The inputs are the first frame and its face bounding box
                else:
                    xx1, yy1, xx2, yy2 = face_localtion_match(
                        np.copy(frame), xx1, yy1, xx2, yy2
                    )  # The inputs are the current frame and the face bounding box of the previous frame
                dir = os.path.relpath(path, jpg_dir)
                txt.write('%s %s %d %d %d %d %d.jpg\n' % (dir, name, xx1, yy1, xx2, yy2, n))
                n += 1
            print('%.3fs' % (time.time()-t))
    txt.close()
