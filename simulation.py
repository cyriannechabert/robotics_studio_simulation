import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
xml_path = "my_robot.mjcf"
SIM_DURATION = 15.0  # Run a bit longer to enjoy the walk

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# --- TUNED GAIT PARAMETERS ---
# REDUCED: Lowered from 0.6 to 0.35 based on your 2-second observation
HIP_AMP = 0.35       
HIP_OFFSET = 0.0     

# SAME: Keep these as they were working well
KNEE_BASE = -0.3   
KNEE_AMP = 0.5     
FREQ = 3.0       

# --- DATA LOGGING ---
log_time = []
log_speed = []
log_motor_pos = []  
log_motor_vel = []  
log_motor_torque = [] 

print("Starting Simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 1. Reset to safe start
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.3 
    mujoco.mj_forward(model, data)

    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # Stop after SIM_DURATION
        if t > SIM_DURATION:
            print("Simulation complete. Closing...")
            viewer.close()
            break

        # --- GAIT LOGIC ---
        # We keep the ramp_up, but now it ramps up to a smaller maximum (0.35)
        ramp_up = min(1.0, t / 3.0) 

        phase_a = np.sin(2 * np.pi * FREQ * t)
        phase_b = np.sin(2 * np.pi * FREQ * t + np.pi)

        current_hip_offset = HIP_OFFSET * ramp_up
        current_knee_base = KNEE_BASE * ramp_up
        
        # Apply to Hips (Smaller steps now!)
        data.ctrl[0] = current_hip_offset + (HIP_AMP * phase_a * ramp_up) 
        data.ctrl[3] = current_hip_offset + (HIP_AMP * phase_a * ramp_up) 
        data.ctrl[1] = current_hip_offset + (HIP_AMP * phase_b * ramp_up) 
        data.ctrl[2] = current_hip_offset + (HIP_AMP * phase_b * ramp_up) 

        # Apply to Knees
        data.ctrl[4] = current_knee_base + (KNEE_AMP * max(0, phase_b) * ramp_up)
        data.ctrl[7] = current_knee_base + (KNEE_AMP * max(0, phase_b) * ramp_up)
        data.ctrl[5] = current_knee_base + (KNEE_AMP * max(0, phase_a) * ramp_up)
        data.ctrl[6] = current_knee_base + (KNEE_AMP * max(0, phase_a) * ramp_up)

        # Step Physics
        mujoco.mj_step(model, data)
        viewer.sync()

        # Logging
        log_time.append(t)
        current_speed = np.linalg.norm(data.qvel[0:3])
        log_speed.append(current_speed)
        
        if model.nu > 0:
            log_motor_pos.append(data.qpos[7:15].copy()) 
            log_motor_vel.append(data.qvel[6:14].copy()) 
            log_motor_torque.append(data.ctrl.copy())

        # Real-time sync
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# --- POST-PROCESSING ---
print("-" * 30)
avg_speed = np.mean(log_speed)
print(f"CALCULATED AVERAGE SPEED: {avg_speed:.4f} m/s")
print("-" * 30)

# Generate Plots
time_arr = np.array(log_time)
pos_arr = np.array(log_motor_pos)
vel_arr = np.array(log_motor_vel)
tor_arr = np.array(log_motor_torque)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 12))
axs[0].plot(time_arr, pos_arr)
axs[0].set_title('Motor Angles (Rad) vs Time')
axs[0].set_ylabel('Position (rad)')
axs[0].grid(True)

axs[1].plot(time_arr, vel_arr)
axs[1].set_title('Motor Velocity (Rad/s) vs Time')
axs[1].set_ylabel('Velocity (rad/s)')
axs[1].grid(True)

axs[2].plot(time_arr, tor_arr)
axs[2].set_title('Motor Commands vs Time')
axs[2].set_ylabel('Command')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)

print("Displaying plots...")
plt.tight_layout()
plt.show()