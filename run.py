import argparse
import json
import logging
import time
from time import sleep

import cv2
import numpy as np
import paramiko
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
    def __init__(self, camera_num=0, w=432, h=368, model='mobilenet_thin', resize_out_ratio=4.0):
        # Camera
        self.camera_num = camera_num
        self.cam = None
        self._init_cam()

        # Model
        self.model = model
        self.resize_out_ratio = resize_out_ratio
        self.w = w
        self.h = h
        self.e = None
        self._init_estimator()

        # SSH with drone
        self.ssh = None
        self.host_keys = '/Users/rshmyrev/.ssh/known_hosts'
        self.drone_ip = '192.168.1.31'
        self.drone_username = 'pi'
        self.drone_pass = 'raspberry'
        # self.init_ssh()

        # Image, humans, body parts
        self.image = None
        self.humans = []
        self.human = None
        self.hand = {'right': [], 'left': []}
        self.elbow_angle = {'right': None, 'left': None}
        self.wrist_position = {'right': None, 'left': None}
        self.hand_direction = {'right': None, 'left': None}
        self.pose = None

        # Draw params
        self.time = time.time()


        # Poses dict
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

    def init_ssh(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.load_host_keys(self.host_keys)
        self.ssh.connect(self.drone_ip, username=self.drone_username, password=self.drone_pass)

    def reset_params(self):
        self.image = None
        self.humans = []
        self.human = None
        self.pose = None
        for hand_name in ('right', 'left'):
            self.hand[hand_name] = None
            self.elbow_angle[hand_name] = None
            self.wrist_position[hand_name] = None
            self.hand_direction[hand_name] = None

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
        """It's you"""
        if self.humans:
            self.human = self.humans[0]  # by Dima

    def destroy_all_humans(self):
        # TODO
        pass

    def get_hands(self):
        if not self.human:
            return

        self.hand['right'] = [(self.human.body_parts[i].x, self.human.body_parts[i].y) for i in range(2, 5) if
                              i in self.human.body_parts]
        self.hand['left'] = [(self.human.body_parts[i].x, self.human.body_parts[i].y) for i in range(5, 8) if
                             i in self.human.body_parts]

        if len(self.hand['right']) != 3:
            self.hand['right'] = []
        if len(self.hand['left']) != 3:
            self.hand['left'] = []

    @staticmethod
    def _elbow_angle(hand):
        """

        :param hand: 3 points: (x1, x2, x3). x2 - координаты вершины угла
        :return: degrees for x1-x2-x3 angle
        """
        x1, x2, x3 = hand

        def length_line(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        p12 = length_line(x1, x2)
        p13 = length_line(x1, x3)
        p23 = length_line(x2, x3)

        return np.degrees(np.arccos((p12 ** 2 - p13 ** 2 + p23 ** 2) / (2 * p12 * p23)))

    @staticmethod
    def _wrist_position(hand):
        x1, x2, x3 = hand
        if x3[1] > x1[1] and x3[1] > x2[1]:
            return 'up'
        elif x3[1] < x1[1] and x3[1] < x2[1]:
            return 'down'
        else:
            return ''

    def _hand_direction(self, hand_name):
        angle_gap = 25  # degree

        angle = self.elbow_angle[hand_name]
        wrist_position = self.wrist_position[hand_name]

        if 90 - angle_gap <= angle <= 90 + angle_gap:
            if wrist_position == 'up':
                return '90'
            elif wrist_position == 'down':
                return '270'

        elif 180 - angle_gap <= angle <= 0 + angle_gap:
            return '180'

        else:
            return ''

    def calculate_hands_position(self):
        for hand_name in ('right', 'left'):
            if not self.hand[hand_name]:
                continue

            self.elbow_angle[hand_name] = self._elbow_angle(self.hand[hand_name])
            self.wrist_position[hand_name] = self._wrist_position(self.hand[hand_name])
            self.hand_direction[hand_name] = self._hand_direction(hand_name)

            logger.debug('{} hand. Angle: {}, wrist_direction: {}, hand_direction: {}'.format(
                hand_name,
                self.elbow_angle[hand_name],
                self.wrist_position[hand_name],
                self.hand_direction[hand_name]
            ))

    def calculate_pose(self):
        if not self.hand_direction['right'] or not self.hand_direction['left']:
            self.pose = None

        if self.hand_direction['right'] == '90' and self.hand_direction['left'] == '90':
            self.pose = 1

        elif self.hand_direction['right'] == '270' and self.hand_direction['left'] == '270':
            self.pose = 2

        elif self.hand_direction['right'] == '180' and self.hand_direction['left'] == '180':
            self.pose = 3

        elif self.hand_direction['right'] == '90' and self.hand_direction['left'] == '180':
            self.pose = 4

        elif self.hand_direction['left'] == '90' and self.hand_direction['right'] == '180':
            self.pose = 5

        elif self.hand_direction['right'] == '270' and self.hand_direction['left'] == '180':
            self.pose = 6

        elif self.hand_direction['left'] == '270' and self.hand_direction['right'] == '180':
            self.pose = 7

    def infinite_loop(self):
        while True:
            self.reset_params()
            self.get_humans()
            self.choose_best_human()
            self.get_hands()
            self.calculate_hands_position()
            self.calculate_pose()
            if self.pose:
                logger.info('Pose {}'.format(self.pose))

            # Draw
            image = TfPoseEstimator.draw_humans(self.image, [self.human, ], imgcopy=False)
            # resize
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

            # Draw angles and pose
            image = self._draw_angle(image)
            image = self._draw_pose(image)
            image = self._draw_fps(image)

            cv2.imshow('Result', image)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def _draw_angle(self, npimg):
        image_h, image_w = npimg.shape[:2]

        for hand_name in ('right', 'left'):
            if not self.elbow_angle[hand_name]:
                continue

            center_point = self.hand[hand_name][1]
            center_on_image = (int(center_point[0] * image_w + 3.5), int(center_point[1] * image_h + 0.5))
            cv2.putText(npimg,
                        "Angle: {}".format(int(self.elbow_angle[hand_name])),
                        center_on_image, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        return npimg

    def _draw_pose(self, npimg):
        if self.pose:
            cv2.putText(npimg,
                        "Pose: %s" % self.pose,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        return npimg

    def _draw_fps(self, npimg):
        cv2.putText(npimg,
                    "FPS: %f" % (1.0 / (time.time() - self.time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        self.time = time.time()
        return npimg

    def pose2commands(self):
        pass

    def commands2route(self):
        pass

    def send2raspberry(self, command, params):
        # 'source /opt/ros/kinetic/setup.bash'
        # 'source /home/pi/catkin_ws/devel/setup.bash'
        # cmd = 'source /opt/ros/kinetic/setup.bash; source /home/pi/catkin_ws/devel/setup.bash; rosservice call /get_telemetry "{frame_id: }"'
        cmd = 'source /opt/ros/kinetic/setup.bash; source /home/pi/catkin_ws/devel/setup.bash; rosservice call /{} "{}"'.format(
            command, json.dumps(params))
        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(cmd)

        return ssh_stdout.read()

    def get_telemetry(self, frame_id=''):
        command = "get_telemetry"
        params = {'frame_id': frame_id}

        ssh_stdout = self.send2raspberry(command, params)
        logger.info(ssh_stdout)

        # channel = self.ssh.get_transport().open_session()
        # channel.get_pty()
        # channel.settimeout(10)
        # channel.exec_command(cmd)
        # ssh_stdout = channel.recv(1024)
        # channel.close()
        # logger.info(ssh_stdout)

        return ssh_stdout

    def navigate(self, x=0, y=0, z=0, speed=0.5, frame_id='aruco_map', update_frame=True, auto_arm=True):
        command = "navigate"
        params = {
            'x': x,
            'y': y,
            'z': z,
            'speed': speed,
            'frame_id': frame_id,
            'update_frame': update_frame,
            'auto_arm': auto_arm,
        }

        ssh_stdout = self.send2raspberry(command, params)
        logger.info(ssh_stdout)

    def square(self, z=1, speed=1, sleep_time=1, update_frame=False):
        self.navigate(x=1, y=1, z=z, speed=speed, frame_id='aruco_map', update_frame=update_frame)
        sleep(sleep_time)
        self.navigate(x=1, y=2, z=z, speed=speed, frame_id='aruco_map', update_frame=update_frame)
        sleep(sleep_time)
        self.navigate(x=2, y=2, z=z, speed=speed, frame_id='aruco_map', update_frame=update_frame)
        sleep(sleep_time)
        self.navigate(x=2, y=1, z=z, speed=speed, frame_id='aruco_map', update_frame=update_frame)
        sleep(sleep_time)
        self.navigate(x=1, y=1, z=z, speed=speed, frame_id='aruco_map', update_frame=update_frame)
        sleep(sleep_time)
        self.land()

    def both_up(self, z=1, tolerance=0.2):
        """
        Up on z metres

        :param z: высота
        :param tolerance: точность проверки высоты (м)
        """
        start = self.get_telemetry()  # Запоминаем изначальную точку
        self.navigate(z=z, speed=0.5, frame_id='aruco_map', auto_arm=True)  # Взлетаем на 2 м
        while True:  # Ожидаем взлета
            if self.get_telemetry().z - start.z + z < tolerance:  # Проверяем текущую высоту
                break
            sleep(0.2)  # ??? как зависание сделать

    def land(self):
        self.send2raspberry('land', params={})


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
    dance.infinite_loop()
    # while True:
    #     dance.get_telemetry()
    #     sleep(1)

    # dance.square()

    # navigate(x=1, y=1, z=1, speed=1, frame_id='aruco_map', update_frame=True)
    # dance.navigate(x=3, y=3, z=2, speed=speed, frame_id='aruco_map', update_frame=True)
