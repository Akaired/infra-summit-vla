# sim/viewer.py
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path

# Загружаем конфиг и сцену
script_dir = Path(__file__).parent
config_path = script_dir.parent / "configs" / "sim.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

mjcf_path = script_dir.parent / config["scene"]["mjcf"]
model = mujoco.MjModel.from_xml_path(str(mjcf_path))
data = mujoco.MjData(model)

# Запускаем интерактивный просмотр
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()