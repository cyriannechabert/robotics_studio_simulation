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

# 2. SAFER GAIT FUNCTION
def get_action(time, params):
    actions = []
    w = 3.0 # Lowered from 4.0 to 3.0 for stability
    
    for i in range(num_actuators):
        a = params[i*3 + 0] 
        b = params[i*3 + 1]
        c = params[i*3 + 2]
        
        theta = a + b * np.sin(w * time + c)
        
        # Double safety: Clip to safe range (-1.2 to 1.2)
        theta = np.clip(theta, -1.2, 1.2)
        
        actions.append(theta)
        
    return np.array(actions)

# 3. EVALUATION LOOP
def evaluate_gait(params, duration=5.0):
    mujoco.mj_resetData(model, data)
    steps = int(duration / model.opt.timestep)
    
    for _ in range(steps):
        ctrl = get_action(data.time, params)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        
        # Fail instantly if physics is unstable
        if not np.all(np.isfinite(data.qpos)):
            return -1000.0
            
    # Measure distance moved in X
    distance = data.body('body_base').xpos[0] 
    return distance

# 4. CONSTRAINED RANDOM SEARCH
best_score = -np.inf
best_params = None

print("Starting Safe Random Search...")

for i in range(1000):
    random_params = []
    for _ in range(num_actuators):
        # --- KEY FIX: SMART GENERATION ---
        
        # 1. Pick a modest Amplitude (b) first (0.1 to 0.5 rads)
        #    This prevents legs from swinging wildly 
        b = np.random.uniform(0.1, 0.5)
        
        # 2. Pick an Offset (a) that fits with the Amplitude
        #    We want abs(a) + b < 1.2 (Safe Limit)
        #    So 'a' must be between -(1.2 - b) and (1.2 - b)
        safe_limit = 1.2 - b
        a = np.random.uniform(-safe_limit, safe_limit)
        
        # 3. Phase (c) can be anything
        c = np.random.uniform(0.0, 2*np.pi)
        
        random_params.extend([a, b, c])
        
    random_params = np.array(random_params)
    
    # Evaluate
    score = evaluate_gait(random_params)
    
    if score > best_score:
        best_score = score
        best_params = random_params
        print(f"Iteration {i}: New Best! Dist: {best_score:.3f} m")

print("Training finished.")

# 5. VISUALIZE
if best_params is not None:
    print("Visualizing best gait... (Press ESC to exit)")
    mujoco.mj_resetData(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            data.ctrl[:] = get_action(data.time, best_params)
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
else:
    print("Optimization failed.")