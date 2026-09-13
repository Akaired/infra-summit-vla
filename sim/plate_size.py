import mujoco

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")

# Найди индекс тела "plate"
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

# Посмотри все геометрии этого тела — их радиусы (rbound) и реальные размеры
for gid in range(model.ngeom):
    if model.geom_bodyid[gid] == bid:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        rbound = model.geom_rbound[gid]  # максимальный радиус от центра
        size = model.geom_size[gid]
        print(f"{name}: rbound={rbound:.4f} м, size={size}")