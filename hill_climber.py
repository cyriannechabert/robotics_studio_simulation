import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. SETUP
model = mujoco.MjModel.from_xml_path('my_robot.mjcf')
data = mujoco.MjData(model)

actuator_names = [
    "hip_1_servo", "knee_1_servo",
    "hip_3_servo", "knee_3_servo",
    "hip_5_servo", "knee_5_servo",
    "hip_7_servo", "knee_7_servo"
]
num_actuators = len(actuator_names)

# 2. GAIT FUNCTION (With clipped limits for safety)
def get_action(time, params):
    actions = []
    w = 3.0 # Slower frequency = less jumping
    
    for i in range(num_actuators):
        a = params[i*3 + 0] 
        b = params[i*3 + 1]
        c = params[i*3 + 2]
        
        theta = a + b * np.sin(w * time + c)
        theta = np.clip(theta, -1.0, 1.0) # Tighter limits to prevent high kicks
        actions.append(theta)
        
    return np.array(actions)

# 3. EVALUATION LOOP (With Stability Checks)
def evaluate_gait(params, duration=5.0):
    mujoco.mj_resetData(model, data)
    steps = int(duration / model.opt.timestep)
    total_energy_cost = 0
    
    for _ in range(steps):
        ctrl = get_action(data.time, params)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        
        # --- STABILITY CHECK 1: EXPLOSION ---
        if not np.all(np.isfinite(data.qpos)):
            return -1000.0
            
        # --- STABILITY CHECK 2: FALLING ---
        # If torso height (z-axis) drops below 0.15m, robot has fallen.
        z_height = data.body('body_base').xpos[2]
        if z_height < 0.15: 
            return -50.0 # Huge penalty for falling
            
        # --- STABILITY CHECK 3: ENERGY ---
        # Penalize large, violent movements (sum of squared commands)
        total_energy_cost += np.sum(np.square(ctrl))

    # Calculate final score
    distance = data.body('body_base').xpos[0]
    
    # Score = Distance - (Energy used * small weight)
    # This encourages "efficient" movement over "violent" movement
    score = distance - (total_energy_cost * 0.001)
    
    return score

# 4. HILL CLIMBER (Evolution)
# Start with a safe "Standing" pose (Offsets=0, Amplitudes=0.1)
current_params = []
for _ in range(num_actuators):
    current_params.extend([0.0, 0.1, 0.0]) 
current_params = np.array(current_params)

best_score = evaluate_gait(current_params)
print(f"Baseline Score (Standing): {best_score:.3f}")
print("Starting Hill Climber...")

# Try to improve 500 times
for i in range(500): 
    # 1. Mutate: Add small changes to our best params
    # We use small noise (std dev 0.05) so it doesn't "jump" to crazy values
    noise = np.random.normal(0, 0.05, size=current_params.shape)
    candidate_params = current_params + noise
    
    # 2. Evaluate
    score = evaluate_gait(candidate_params)
    
    # 3. Selection
    if score > best_score:
        print(f"Iter {i}: Improved! Score: {best_score:.3f} -> {score:.3f}")
        best_score = score
        current_params = candidate_params # Adopt the new gene
    # If not better, we just discard 'candidate_params' and try again

print("Training finished.")

# 5. VISUALIZE
print("Visualizing best walker... (Press ESC to exit)")
mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        data.ctrl[:] = get_action(data.time, current_params)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)