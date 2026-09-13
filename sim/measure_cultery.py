import mujoco
import numpy as np

xml = """
<mujoco>
  <compiler meshdir="sim/assets/ycb" texturedir="sim/assets/ycb"/>
  <asset>
    <mesh name="fork_visual_mesh" file="fork/textured.obj"/>
    <mesh name="fork_collision_mesh_0" file="fork/textured_coacd_0.stl"/>
    <mesh name="fork_collision_mesh_1" file="fork/textured_coacd_1.stl"/>
    <texture name="fork_texture" type="2d" file="fork/texture_map.png"/>
    <material name="fork_material" texture="fork_texture"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 0 1.6"/>
    <body name="table" pos="0 0 0">
      <geom type="box" size="0.5 0.35 0.02" pos="0 0 0.75" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="fork" pos="0 0 0.8" euler="-1.5708 0 0">
      <joint type="free" damping="0.001"/>
      <geom type="mesh" mesh="fork_visual_mesh" material="fork_material" contype="0" conaffinity="0" group="0"/>
      <geom type="mesh" mesh="fork_collision_mesh_0" contype="1" conaffinity="1" group="3"/>
      <geom type="mesh" mesh="fork_collision_mesh_1" contype="1" conaffinity="1" group="3"/>
      <inertial pos="0 0 0" mass="0.034" diaginertia="0.001 0.001 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

print("Загружаем модель...")
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Симулируем до успокоения...")
for i in range(2000):
    mujoco.mj_step(model, data)
    if i % 100 == 0:
        data.qvel[:] = 0.0

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fork")
pos = data.xpos[body_id]
print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Вилка успокоилась на z = {pos[2]:.4f} м")

print("\nРазмеры геометрий (bounding box):")
for gid in range(model.ngeom):
    if model.geom_bodyid[gid] == body_id:
        size = model.geom_size[gid]
        print(f"  Геометрия {gid}: size = {size}")