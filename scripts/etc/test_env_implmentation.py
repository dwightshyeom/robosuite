import robosuite as suite
import numpy as np
import imageio

# 1. Create the environment
env = suite.make(
    env_name="Wipe",                # Task name
    robots="Panda",                 # Robot to use
    has_renderer=True,             # Set to True if you want a pop-up window (onscreen)
    has_offscreen_renderer=True,    # Must be True to capture video frames
    use_camera_obs=True,            # We need camera observations
    camera_names="frontview",       # The camera angle to record
    camera_heights=512,             # Video resolution height
    camera_widths=512,              # Video resolution width
)

# 2. Reset the environment
obs = env.reset()

frames = []

# 3. Step through the simulation
for i in range(10000):
    # Take a random action (replace this with your policy's output)
    action = np.random.randn(6)/1000
    obs, reward, done, info = env.step(action)
    
    # 4. Extract the image frame from the observation dictionary
    # Robosuite returns images upside down by default, so we flip it
    frame = obs["frontview_image"]
    frame = np.flipud(frame)
    frames.append(frame)

# # 5. Save the frames to an MP4 video
# writer = imageio.get_writer("my_simulation_video.mp4", fps=20)
# for frame in frames:
#     writer.append_data(frame)
# writer.close()

print("Video saved successfully!")