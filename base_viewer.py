import mujoco
import mujoco.viewer
import time

# Define the file path for your MuJoCo model file (in XML format).
from robot_descriptions import mujoco_humanoid_mj_description


xml_path= "my_robot.mjcf"
# Load the model from the specified XML file.
#This model object contains the static, structural definition of the simulation (e.g., masses, joint types, collision geometries).
model = mujoco.MjModel.from_xml_path(xml_path)
# The MjData object contains the dynamic state of the simulation: positions, velocities, forces, sensor data, etc.
data = mujoco.MjData(model)
# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Run the simulation loop as long as the viewer window is open.
    while viewer.is_running():
        step_start = time.time()
        # Step the simulation
        mujoco.mj_step(model, data)
        # Update the viewer's display with the new state from the MjData object.
        viewer.sync()
        # Optional: To maintain a real-time simulation speed.
        # model.opt.timestep is the desired duration of one step (from the XML).
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



