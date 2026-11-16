# Kinova pick-and-place demo

Guide a simulated Kinova Gen3 arm by applying a force while it carries a cube.

## Run

From this folder, create the environment once and start the simulator.

```bash
conda create -n kinova -c conda-forge python=3.10 pybullet numpy
conda activate kinova
python src/keyboard_teleop.py
```

## Try a push

1. Focus the simulation window, press `F`, and click the forearm or wrist.
   A cyan marker follows the selected point.
2. Press `P`. When the status says **carrying**, hold `D` for a lateral
   push. The force readout shows **6 N** while you hold the key.
3. Release the key. The arm returns to the paused waypoint and finishes
   placing the cube.

Use `W/S`, `A/D`, and `Q/E` for pushes along ±x, ±y, and ±z.
Press `F` again for manual movement or `P` to stop the task.

## Manual controls

| Keys | Action |
| --- | --- |
| `W/S`, `A/D`, `Q/E` | Move along x, y, z |
| `U/O`, `I/K`, `J/L` | Rotate about x, y, z |
| `M` | Switch between Cartesian and joint mode. In joint mode, pick `1–7` and move with `W/S` or ↑/↓. |
| `Space`, `[`, `]` | Toggle the gripper, or hold a bracket to open or close it |
| `H`, `-` / `=`, `X` | Home, change manual speed, quit |

[Admittance equations and ME 531 references](ADMITTANCE.md)

Run the headless tests with `python tests/test_teleop.py`.

Robot meshes and URDF are from `afs_robot_exp/robots/kinova`, with gripper
joints and inertias adapted for PyBullet.
