rosservice call /finish_trajectory 0
sleep 2
rosservice call /write_state "{filename: '/home/rc1/map20250805.pbstream', include_unfinished_submaps: true}"