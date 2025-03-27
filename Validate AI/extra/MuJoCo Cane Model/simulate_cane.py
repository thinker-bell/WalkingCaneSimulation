import mujoco
import mujoco.viewer
import numpy as np

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("MuJoCo Cane Model\cane.xml")
data = mujoco.MjData(model)

# Run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Apply swing motion (alternating torques)
        data.ctrl[0] = np.sin(data.time * 2)  # Sine wave for smooth oscillation

        # Update viewer
        viewer.sync()
