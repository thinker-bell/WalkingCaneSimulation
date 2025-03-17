import pybullet as p
import pybullet_data
import time
import math

def run_lidar_simulation():
    """
    Simulates a robot with a lidar sensor in PyBullet, visualizing the lidar and moving the robot.
    """

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    planeId = p.loadURDF("plane.urdf")
    robotStartPos = [0, 0, 0.1]
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF("r2d2.urdf", robotStartPos, robotStartOrientation)

    num_rays = 30  # Reduced number of rays to focus the scan
    ray_length = 5
    scan_angle = math.pi / 4  # Scan angle of 45 degrees in front of the robot

    for i in range(1000):
        p.stepSimulation()

        # Update lidar ray origins and destinations
        ray_froms = []
        ray_tos = []
        base_pos, base_orient = p.getBasePositionAndOrientation(robotId)
        yaw = p.getEulerFromQuaternion(base_orient)[2]  # Get the robot's yaw angle
        for j in range(num_rays):
            angle = -scan_angle / 2 + j * scan_angle / (num_rays - 1)
            rotated_vector = [ray_length * math.cos(yaw + angle), ray_length * math.sin(yaw + angle), 0]
            ray_from = base_pos
            ray_to = [base_pos[0] + rotated_vector[0], base_pos[1] + rotated_vector[1], base_pos[2]]
            ray_froms.append(ray_from)
            ray_tos.append(ray_to)

        ray_results = p.rayTestBatch(ray_froms, ray_tos)

        obstacles_detected = False
        for j, ray_result in enumerate(ray_results):
            if ray_result[0] != -1:
                obstacles_detected = True
                p.addUserDebugLine(ray_froms[j], ray_result[3], [1, 0, 0], 1)
            else:
                p.addUserDebugLine(ray_froms[j], ray_tos[j], [0, 1, 0], 1)

        if not obstacles_detected:
            targetVelocity = 1  # Reduced target velocity
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=targetVelocity)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, targetVelocity=targetVelocity)
        else:
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=0)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, targetVelocity=0)
            print("Obstacle Detected!")

        time.sleep(1. / 60.)  # Increased sleep time to slow down the simulation

    p.disconnect()

if __name__ == "__main__":
    run_lidar_simulation()