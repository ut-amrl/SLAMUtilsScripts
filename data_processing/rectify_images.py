# system
import os 
import argparse
import shutil

# images
import cv2
import numpy as np

kImgWidth = 960
kImgHeight = 600

kKLeft = np.array([[527.873518, 0.000000, 482.823413],
                   [0.000000, 527.276819, 298.033945], 
                   [0.000000, 0.000000, 1.000000]])
kDLeft = np.array([0, 0, 0, 0, 0])
kRLeft = np.array([[0.999940, -0.003244, -0.010471], 
                   [0.003318, 0.999970, 0.007064], 
                   [0.010448, -0.007098, 0.999920]])
kPLeft = np.array([[528.955512, 0.000000, 479.748173, 0.000000], 
                   [0.000000, 528.955512, 298.607571, 0.000000], 
                   [0.000000, 0.000000, 1.000000, 0.000000]])

kKRight = np.array([[530.158021, 0.000000, 475.540633], 
                    [0.000000, 529.682234, 299.995465], 
                    [0.000000, 0.000000, 1.000000]])
kDRight = np.array([0, 0, 0, 0, 0])
kRRight = np.array([[0.999661, -0.024534, 0.008699], 
                    [0.024595, 0.999673, -0.006974], 
                    [-0.008525, 0.007186, 0.999938]])
kPRight = np.array([[528.955512, 0.000000, 479.748173, -69.690815], 
                    [0.000000, 528.955512, 298.607571, 0.000000], 
                    [0.000000, 0.000000, 1.000000, 0.000000]])

kMapLeft1, kMapLeft2 = cv2.initUndistortRectifyMap(kKLeft, kDLeft, kRLeft, kPLeft, (kImgWidth, kImgHeight), cv2.CV_32F)
kMapRight1, kMapRight2 = cv2.initUndistortRectifyMap(kKRight, kDRight, kRRight, kPRight, (kImgWidth, kImgHeight), cv2.CV_32F)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="", required=True, type=str)
    parser.add_argument("--outputdir", default="", required=True, type=str)
    args = parser.parse_args()
    if not os.path.exists(args.inputdir):
        raise FileNotFoundError("Input root directory " + args.inputdir + " doesn't exist")
    if not os.path.exists(args.outputdir):
        print("Creating output directory " + args.outputdir)
        os.makedirs(args.outputdir, exist_ok=True)
    return args

def rectify_kitti_stereo(inputdir: str, outputdir: str):
    inputdir_cam0 = os.path.join(inputdir, "image_0")
    if not os.path.exists(inputdir_cam0):
        raise FileNotFoundError("Input directory " + inputdir_cam0 + " doesn't exist")
    inputdir_cam1 = os.path.join(inputdir, "image_1")
    if not os.path.exists(inputdir_cam1):
        raise FileNotFoundError("Input directory " + inputdir_cam1 + " doesn't exist")
    outputdir_cam0 = os.path.join(outputdir, "image_0")
    if not os.path.exists(outputdir_cam0):
        print("Creating output directory " + outputdir_cam0)
        os.mkdir(outputdir_cam0)
    outputdir_cam1 = os.path.join(outputdir, "image_1")
    if not os.path.exists(outputdir_cam1):
        print("Creating output directory " + outputdir_cam1)
        os.mkdir(outputdir_cam1)
    for filename in os.listdir(inputdir_cam0):
        img = cv2.imread(os.path.join(inputdir_cam0, filename))
        img = cv2.remap(img, kMapLeft1, kMapLeft2, cv2.INTER_LINEAR)
        outputpath = os.path.join(outputdir_cam0, filename)
        cv2.imwrite(outputpath, img)
    for filename in os.listdir(inputdir_cam1):
        img = cv2.imread(os.path.join(inputdir_cam1, filename))
        img = cv2.remap(img, kMapRight1, kMapLeft2, cv2.INTER_LINEAR)
        outputpath = os.path.join(outputdir_cam1, filename)
        cv2.imwrite(outputpath, img)
    shutil.copy(os.path.join(inputdir, "times.txt"), os.path.join(outputdir, "times.txt"))

if __name__ == "__main__":
    args = parse_opt()
    print(args)
    rectify_kitti_stereo(args.inputdir, args.outputdir)