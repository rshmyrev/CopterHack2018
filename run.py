import argparse
import logging
import time

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

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
        self.e = None

        self.image = None
        self.humans = list()

        self._init_cam()
        self._init_estimator()

        self.poses = {
            1: 'both up',
            2: 'both down',
            3: 'both side',
            4: 'right up, left side',
            5: 'left up, right side',
            6: 'right down, left side',
            7: 'left down, right side',
        }

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
        # logger.debug('Session: {}'.format(self.e.persistent_sess))
        logger.debug('Count of humans: {}'.format(len(self.humans)))
        # logger.debug('Humans parts: {}'.format(self.humans[0]))

    def choose_best_human(self):
        "It's you"
        return self.humans[0]  # by Dima

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

    @staticmethod
    def get_hands(human):
        right_hand = [(human.body_parts[i].x, human.body_parts[i].y) for i in range(2, 5) if i in human.body_parts]
        left_hand = [(human.body_parts[i].x, human.body_parts[i].y) for i in range(5, 8) if i in human.body_parts]

        if len(right_hand) != 3:
            right_hand = None
        if len(left_hand) != 3:
            left_hand = None

        return right_hand, left_hand

    @staticmethod
    def elbow_angle(hand):
        x1, x2, x3 = hand

        def length_line(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        p12 = length_line(x1, x2)  # x2 - координаты вершины угла
        p13 = length_line(x1, x3)
        p23 = length_line(x2, x3)

        return np.degrees(np.arccos((p12 ** 2 - p13 ** 2 + p23 ** 2) / (2 * p12 * p23)))

    def parts2pose(self, right_hand, left_hand):
        # TODO: something wrong with left wrist
        def wrist_position(hand):
            x1, x2, x3 = hand
            if x3 > x1 and x3 > x2:
                return 'up'
            elif x3 < x1 and x3 < x2:
                return 'down'
            else:
                return ''

        def hand_direction(hand):
            angle_gap = 25  # degree

            if 90 - angle_gap <= self.elbow_angle(hand) <= 90 + angle_gap:
                if wrist_position(hand) == 'up':
                    return '90'
                elif wrist_position(hand) == 'down':
                    return '270'

            elif 180 - angle_gap <= self.elbow_angle(hand) <= 0 + angle_gap:
                return '180'

            else:
                return ''

        if not right_hand or not left_hand:
            return None

        logger.debug('Right hand. Angle: {}, wrist_direction: {}, hand_direction: {}'.format(
            self.elbow_angle(right_hand),
            wrist_position(right_hand),
            hand_direction(right_hand)
        ))
        logger.debug('Left hand. Angle: {}, wrist_direction: {}, hand_direction: {}'.format(
            self.elbow_angle(left_hand),
            wrist_position(left_hand),
            hand_direction(left_hand)
        ))

        rh_dir = hand_direction(right_hand)
        lh_dir = hand_direction(left_hand)

        if rh_dir == '90' and lh_dir == '90':
            return 1

        elif rh_dir == '270' and lh_dir == '270':
            return 2

        elif rh_dir == '180' and lh_dir == '180':
            return 3

        elif rh_dir == '90' and lh_dir == '180':
            return 4

        elif lh_dir == '90' and rh_dir == '180':
            return 5

        elif rh_dir == '270' and lh_dir == '180':
            return 6

        elif lh_dir == '270' and rh_dir == '180':
            return 7

    def infinite_loop(self):
        fps_time = 0

        while True:
            self.get_humans()
            human = self.choose_best_human()
            image = TfPoseEstimator.draw_humans(self.image, [human, ], imgcopy=False)

            right_hand, left_hand = self.get_hands(human)

            right_angle, left_angle = None, None
            if right_hand:
                right_angle = self.elbow_angle(right_hand)

            if left_hand:
                left_angle = self.elbow_angle(left_hand)

            pose = self.parts2pose(right_hand, left_hand)
            logger.info('Pose_num: {}'.format(pose))
            if pose:
                pose = self.poses[pose]
                cv2.putText(image,
                            "Pose: %s" % pose,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            image = self.draw_angle(image, human, right_angle, left_angle)

            # resize
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def draw_angle(self, npimg, human, right_angle, left_angle):
        image_h, image_w = npimg.shape[:2]

        if 3 in human.body_parts.keys() and right_angle:
            body_part = human.body_parts[3]
            center = (int(body_part.x * image_w + 3.5), int(body_part.y * image_h + 0.5))
            cv2.putText(npimg,
                        "Angle: {}".format(right_angle),
                        center, cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        if 6 in human.body_parts.keys() and left_angle:
            body_part = human.body_parts[6]
            center = (int(body_part.x * image_w + 3.5), int(body_part.y * image_h + 0.5))
            cv2.putText(npimg,
                        "Angle: {}".format(left_angle),
                        center, cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        return npimg

    def angle(self):
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

    dance = Dance(camera_num=args.camera, model=args.model)
    # dance.infinite_cam()
    dance.infinite_loop()
