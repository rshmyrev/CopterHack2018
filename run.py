import argparse
import logging
import time

import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Dance:
    def __init__(self, camera_num=0, w=432, h=368, resize_out_ratio=4.0, model='mobilenet_thin'):
        self.camera_num = camera_num
        self.cam = None
        self.model = model
        self.resize_out_ratio = resize_out_ratio
        self.w = w
        self.h = h
        # self.e = None

        self.image = None
        self.humans = list()

        self._init_cam()
        self._init_estimator()

    def _init_cam(self):
        self.cam = cv2.VideoCapture(self.camera_num)
        self.cam.read()

    def _init_estimator(self):
        logger.debug('initialization %s : %s' % (self.model, get_graph_path(self.model)))
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(self.model), target_size=(self.w, self.h))
        else:
            self.e = TfPoseEstimator(get_graph_path(self.model), target_size=(432, 368))

    def get_humans(self):
        _, self.image = self.cam.read()
        logger.info('cam image={:d}x{:d}'.format(self.image.shape[1], self.image.shape[0]))

        logger.debug('image process+')
        self.humans = self.e.inference(self.image, resize_to_default=(self.w > 0 and self.h > 0),
                                       upsample_size=self.resize_out_ratio)
        logger.debug('Session: {}'.format(self.e.persistent_sess))
        logger.debug('Count of humans: {}'.format(len(self.humans)))
        logger.debug('Humans parts: {}'.format(self.humans[0]))

    def choose_best_human(self, humans):
        "It's you"
        pass

    def destroy_all_humans(self):
        # TODO
        pass

    def infinite_cam(self):
        fps_time = 0
        while True:
            self.get_humans()

            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(self.image, self.humans, imgcopy=False)

            # resize
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')

        cv2.destroyAllWindows()

    def inf_cam(self, args):
        w, h = self.w, self.h
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

        logger.debug('cam read+')
        cam = cv2.VideoCapture(args.camera)
        ret_val, image = cam.read()

        logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

        fps_time = 0

        while True:
            ret_val, image = cam.read()

            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            print('Humans: {}'.format(len(humans)))
            print('Humans[0]: {}'.format(humans[0]))

            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            # resize
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')

        cv2.destroyAllWindows()

    def parts2pose(self):
        pass

    def pose2commands(self):
        pass

    def commands2route(self):
        pass

    def send2raspberry(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    # Dance().inf_cam(args)
    dance = Dance(camera_num=args.camera, model=args.model)
    dance.infinite_cam()
