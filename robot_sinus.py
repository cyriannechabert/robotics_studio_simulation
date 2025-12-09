import numpy as np
import time
import mujoco
import mujoco.viewer

# ... (Load your model/data as before) ...

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


# --- GAIT PARAMETERS (The "Parametric Optimization" variables) ---
# From Slide 7: theta = a + b * sin(omega * t + c)
OMEGA = 4.0        # Speed of the cycle (frequency)
HIP_AMP = 0.6      # (b) How big the step is
HIP_OFFSET = 0     # (a) Center point of the hip
KNEE_AMP = 0.5     # (b) How much the knee bends
KNEE_BASE = -0.5   # (a) Knees should be naturally bent, not straight

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # --- TUNED GAIT PARAMETERS ---
    FREQ = 1.0       # Speed of the step
    
    # Hip: Swings the leg forward/back
    HIP_OFFSET = 0.0     
    HIP_AMP = 0.6        
    
    # Knee: Bends the leg
    # We make KNEE_BASE closer to 0 so the robot stands taller (straighter legs)
    KNEE_BASE = -0.3      
    KNEE_AMP = 0.5        

    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # --- SOFT START LOGIC ---
        # This creates a multiplier that goes from 0.0 to 1.0 over the first 3 seconds.
        # It prevents the "crazy" initial jerk.
        ramp_up = min(1.0, t / 10.0) 

        # --- GAIT GENERATION ---
        # Note: We multiply the AMPLITUDES by 'ramp_up'
        
        # Phase Calculation
        phase_a = np.sin(2 * np.pi * FREQ * t)
        phase_b = np.sin(2 * np.pi * FREQ * t + np.pi)

        # Apply to Hips (Swing)
        # We apply ramp_up to HIP_AMP so it starts with small steps
        data.ctrl[0] = HIP_OFFSET + (HIP_AMP * phase_a * ramp_up) 
        data.ctrl[3] = HIP_OFFSET + (HIP_AMP * phase_a * ramp_up) 
        
        data.ctrl[1] = HIP_OFFSET + (HIP_AMP * phase_b * ramp_up) 
        data.ctrl[2] = HIP_OFFSET + (HIP_AMP * phase_b * ramp_up) 

        # Apply to Knees (Lift)
        # We apply ramp_up to KNEE_AMP so it starts with small lifts
        data.ctrl[4] = KNEE_BASE + (KNEE_AMP * max(0, phase_b) * ramp_up)
        data.ctrl[7] = KNEE_BASE + (KNEE_AMP * max(0, phase_b) * ramp_up)
        
        data.ctrl[5] = KNEE_BASE + (KNEE_AMP * max(0, phase_a) * ramp_up)
        data.ctrl[6] = KNEE_BASE + (KNEE_AMP * max(0, phase_a) * ramp_up)

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Timing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)