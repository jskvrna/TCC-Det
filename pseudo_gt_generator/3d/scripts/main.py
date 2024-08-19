import sys
import argparse
from main_class import MainClass
try:
    import tensorflow.compat.v1 as tf
    # Necessary if gpu is used. Tensorflow with pytorch doesn't like each other as tensorflow loves to allocate all memory for himself :)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Limit TensorFlow GPU memory allocation
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])  # Limit to 0.1GB
        except RuntimeError as e:
            print(e)
except ImportError:
    print("Tensorflow not found. if waymo dataset is to be used, tensorflow is required.")

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Main script for 3D object detection')
    parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'waymo'], help='Dataset to use: kitti or waymo (default: kitti)')
    parser.add_argument('--action', type=str, default='demo', choices=['lidar_scans', 'transformations', 'homographies', 'mask_tracking',
                                 'frames_aggregation', 'optimization', 'demo'], help=('Action to perform: lidar_scans, transformations, homographies, '
                              'mask_tracking, frames_aggregation, optimization, demo (default: demo)'))
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to the configuration file (default: ../configs/config.yaml)')
    parser.add_argument('--seq_start', type=int, default=-1, help='Sequence start index (default: -1 -> use the one in config)')
    parser.add_argument('--seq_end', type=int, default=-1, help='Sequence end index (default: -1 -> use the one in config)')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    autolabel = MainClass(args)

    if autolabel.cfg.frames_creation.tracker_for_merging != '2D' and args.action == 'homographies':
        raise ValueError('Homographies can only be generated with 2D tracker')

    if args.dataset == 'waymo':
        if autolabel.cfg.custom_dataset.use_custom_dataset:
            autolabel.main_custom(args)
        else:
            autolabel.main_waymo(args)
    else:
        autolabel.main_kitti(args)
