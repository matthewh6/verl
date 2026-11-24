This is the recipe's fake simulator (`sim_bridge.py --backend fake`), used by tests and smoke runs only.

The robot has two arms (`left_arm`, `right_arm`: x, y, z in centimeters; roll, pitch, yaw in radians) and two grippers (`left_gripper`, `right_gripper`: width in centimeters). An action gives the change of every quantity. The task is finished by lowering `left_arm` in steps: every action whose `left_arm` z change is -5 or less makes one step of progress, and `left_arm` z shows how far there is still to go.
