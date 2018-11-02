rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 1.5, speed: 1, auto_arm: True, update_frame: False}"
sleep 4
rosservice call /land "{}"