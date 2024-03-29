# system
import os 
import argparse
from pathlib import Path

# images
import cv2

# ros
from cv_bridge import CvBridge
import rosbag

kLeftImageTopicName = "/zed2i/zed_node/left/image_rect_color/compressed"
kRightImageTopicName = "/zed2i/zed_node/right/image_rect_color/compressed"
kLeftCamDirName = "image_0"
kRightCamDirName = "image_1"
kTimestampFilename = "times.txt"
kNodeAndTimestampFilename = "node_ids_and_timestamps.txt"
kImageSuffix = ".png"
kStartIndex = 0

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", default="", required=True, type=str, 
                        help="Bagfile path")
    parser.add_argument("--outputdir", default="", required=True, type=str,
                        help="Output root directory. It will create a subdirectory named by <outputname>.")
    parser.add_argument("--outputname", default="", required=False, type=str,
                        help="Output name. If it's not specified, it will use the bagfile name.")
    parser.add_argument("--left_img_topic", default="", required=False, type=str,
                        help="Left Image ROS Topic name")
    parser.add_argument("--right_img_topic", default="", required=False, type=str,
                        help="Rights Image ROS Topic name")
    args = parser.parse_args()
    if not os.path.exists(args.bagfile):
        raise FileNotFoundError("Input bagfile " + args.bagfile + " doesn't exist")
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)
    if args.outputname == "":
        args.outputname = Path(args.bagfile).stem
        print("You didn't specify otuput name. Current default output name is ", args.outputname)
    args.outputdir = os.path.join(args.outputdir, args.outputname)
    if not os.path.exists(args.outputdir):
        print("Creating output directory " + args.outputdir)
        os.makedirs(args.outputdir, exist_ok=True)
    if args.left_img_topic == "":
        print("Not providing left image topic. Using default")
    else:
        kLeftImageTopicName = args.left_img_topic
    if args.right_img_topic == "":
        print("Not providing right image topic. Using default")
    else:
        kRightImageTopicName = args.right_img_topic
    return args

def parse_bag(args):
    bag = rosbag.Bag(args.bagfile, "r")
    cam0_dir = os.path.join(args.outputdir, kLeftCamDirName)
    cam1_dir = os.path.join(args.outputdir, kRightCamDirName)
    if not os.path.exists(cam0_dir):
        os.mkdir(cam0_dir)
    if not os.path.exists(cam1_dir):
        os.mkdir(cam1_dir)
    timestamp_filepath = os.path.join(args.outputdir, kTimestampFilename)
    fp_timestamp = open(timestamp_filepath, "w")
    if fp_timestamp.closed:
        raise FileNotFoundError("Failed to open file " + timestamp_filepath)
    node_and_timestamp_filepath = os.path.join(args.outputdir, kNodeAndTimestampFilename)
    fp_node_and_timestamp = open(node_and_timestamp_filepath, "w")
    if fp_node_and_timestamp.closed:
        raise FileNotFoundError("Failed to open file " + node_and_timestamp_filepath)
    fp_node_and_timestamp.write("node_id, seconds, nanoseconds\n")

    left_img_idx = kStartIndex
    right_img_idx = kStartIndex
    for topic, msg, t in bag.read_messages(topics=[kLeftImageTopicName, kRightImageTopicName]):
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if topic == kLeftImageTopicName:
            filename = f'{left_img_idx:06}' + kImageSuffix
            filepath = os.path.join(cam0_dir, filename)
            cv2.imwrite(filepath, img)
            fp_timestamp.write(str(t.to_time())+"\n")
            fp_node_and_timestamp.write(str(left_img_idx) + ", " + str(t.secs) + ", " + str(t.nsecs) + "\n")
            left_img_idx += 1
        elif topic == kRightImageTopicName:
            filename = f'{right_img_idx:06}' + kImageSuffix
            filepath = os.path.join(cam1_dir, filename)
            cv2.imwrite(filepath, img)
            right_img_idx += 1
    fp_timestamp.close()
    fp_node_and_timestamp.close()

if __name__ == "__main__":
    args = parse_opt()
    print(args)
    parse_bag(args)
    print("Finish parsing " + args.bagfile)
