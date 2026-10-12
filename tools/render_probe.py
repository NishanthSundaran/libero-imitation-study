"""Render LIBERO-Object task 0 from its fixed init state and report floor colour.
Used to test whether the MuJoCo version changes how textures render (dataset vs env mismatch)."""
import os, sys, numpy as np
from PIL import Image
import mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

b = benchmark.get_benchmark_dict()["libero_object"]()
task = b.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
obs = env.set_init_state(b.get_task_init_states(0)[0])
for _ in range(10):  # same settling steps LeRobot uses before the first observation
    obs, *_ = env.step([0.0] * 6 + [-1.0])
img = obs["agentview_image"][::-1, ::-1]  # rotate 180 deg like the dataset
out = sys.argv[1] if len(sys.argv) > 1 else "probe.png"
Image.fromarray(img).save(out)
f = img.astype(float)
print(f"mujoco {mujoco.__version__}: floor-left {f[200:250,5:60].reshape(-1,3).mean(0).round()} "
      f"floor-right {f[200:250,200:250].reshape(-1,3).mean(0).round()} basket {f[110:150,25:70].reshape(-1,3).mean(0).round()} "
      f"image {f.reshape(-1,3).mean(0).round()}")
