import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. SETUP
model = mujoco.MjModel.from_xml_path('my_robot.mjcf')
data = mujoco.MjData(model)

# --- PHYSICS TUNING ---
# 1. High Gravity to stop jumping
model.opt.gravity[:] = [0, 0, -20.0]
# 2. Smaller timestep for stable math (default is usually 0.002)
model.opt.timestep = 0.002 

# --- INDICES (Corrected) ---
GROUP_A_HIPS  = [0, 3] # Leg 1 & 7
GROUP_A_KNEES = [4, 7]
GROUP_B_HIPS  = [1, 2] # Leg 3 & 5
GROUP_B_KNEES = [5, 6]

# 2. GAIT FUNCTION
def get_action(time, params):
    actions = np.zeros(8)
    w = 2.0 
    
    # Unpack 5 variables
    hip_a, hip_b = params[0], params[1]
    knee_a, knee_b, knee_c = params[2], params[3], params[4]
    
    # --- STABILITY LIMITS ---
    # Force the "Center" (a) to be near 0 so it stays standing
    hip_a = np.clip(hip_a, -0.2, 0.2)
    knee_a = np.clip(knee_a, -0.2, 0.2)
    
    # Limit Amplitude
    hip_b = np.clip(hip_b, 0.0, 0.5)
    knee_b = np.clip(knee_b, 0.0, 0.5)
    
    def get_val(center, amp, phase):
        return np.clip(center + amp * np.sin(w * time + phase), -1.0, 1.0)

    # Group A (Phase 0)
    base_hip  = get_val(hip_a, hip_b, 0)
    base_knee = get_val(knee_a, knee_b, 0 + knee_c)
    
    # Group B (Phase PI)
    opp_hip   = get_val(hip_a, hip_b, np.pi)
    opp_knee  = get_val(knee_a, knee_b, np.pi + knee_c)
    
    # Assign
    for i in GROUP_A_HIPS:  actions[i] = base_hip
    for i in GROUP_A_KNEES: actions[i] = base_knee
    for i in GROUP_B_HIPS:  actions[i] = opp_hip
    for i in GROUP_B_KNEES: actions[i] = opp_knee
        
    return actions

# 3. EVALUATION (With Decimation)
def evaluate_gait(params, duration=6.0):
    mujoco.mj_resetData(model, data)
    
    # PHYSICS SETTINGS
    physics_dt = model.opt.timestep  # e.g. 0.002s
    control_dt = 0.05                # Update motors every 0.05s (20Hz)
    
    # How many physics steps per one control step?
    n_substeps = int(control_dt / physics_dt)
    
    steps = int(duration / control_dt)
    
    for _ in range(steps):
        # 1. Calculate Action ONCE
        ctrl = get_action(data.time, params)
        data.ctrl[:] = ctrl
        
        # 2. Run Physics for N steps with the SAME action
        # This acts like a "Hold" filter, smoothing the motion
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
            
            # Crash Check inside the sub-loop
            if not np.all(np.isfinite(data.qpos)): return -1000.0
            if data.body('body_base').xpos[2] < 0.1: return -50.0

    return data.body('body_base').xpos[0]

# 4. HILL CLIMBER
current_params = np.array([0.0, 0.2, 0.0, 0.2, 1.5]) # Safe start
best_score = evaluate_gait(current_params)
print(f"Baseline: {best_score:.3f}")
print("Starting Stable Training...")

for i in range(500): 
    noise = np.random.normal(0, 0.1, size=current_params.shape)
    candidate_params = current_params + noise
    
    score = evaluate_gait(candidate_params)
    
    if score > best_score:
        print(f"Iter {i}: Improved! {best_score:.2f} -> {score:.2f}")
        best_score = score
        current_params = candidate_params

print("Done.")

# 5. VISUALIZE (With Decimation Logic)
print("Visualizing...")
mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    control_dt = 0.05
    last_control_time = 0
    
    while viewer.is_running():
        # Only update motors if enough time has passed
        if data.time - last_control_time >= control_dt:
            data.ctrl[:] = get_action(data.time, current_params)
            last_control_time = data.time
            
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Check for crashes
        if not np.all(np.isfinite(data.qpos)):
            print("Resetting...")
            mujoco.mj_resetData(model, data)

        # Real-time sleep
        time.sleep(model.opt.timestep)