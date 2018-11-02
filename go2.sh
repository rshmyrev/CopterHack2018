rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 1, speed: 1, auto_arm: True, update_frame: False}"
sleep 3
rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 2, speed: 2, auto_arm: True, update_frame: False}"
sleep 3
rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 1, speed: 2, auto_arm: True, update_frame: False}"
sleep 3
rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 3, speed: 2, auto_arm: True, update_frame: False}"
sleep 3
rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 1, speed: 2, auto_arm: True, update_frame: False}"
sleep 3
rosservice call /navigate "{frame_id: 'aruco_map', x: 3, y: 7, z: 4, speed: 2, auto_arm: True, update_frame: False}"





sleep 4



rosservice call /navigate "{frame_id: 'aruco_map', x: 5, y: 5, z: 1, speed: 1, auto_arm: True, update_frame: False}"
sleep 6
rosservice call /navigate "{frame_id: 'aruco_map', x: 1, y: 5, z: 3, speed: 1, auto_arm: True, update_frame: False}"
sleep 4
rosservice call /navigate "{frame_id: 'aruco_map', x: 1, y: 8, z: 3, speed: 1, auto_arm: True, update_frame: False}"
sleep 4
rosservice call /navigate "{frame_id: 'aruco_map', x: 5, y: 8, z: 1, speed: 1, auto_arm: True, update_frame: False}"
sleep 4





rosservice call /land "{}"
