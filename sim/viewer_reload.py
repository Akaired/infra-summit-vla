# sim/viewer_reload.py
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path
import time

script_dir = Path(__file__).parent
config_path = script_dir.parent / "configs" / "sim.yaml"

def load_model():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mjcf_path = script_dir.parent / config["scene"]["mjcf"]
    print(f"Загружаю XML: {mjcf_path}") # <-- КРИТИЧНО ВАЖНО: проверь, что путь совпадает с тем файлом, который ты редактируешь!
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    # ВАЖНО: Обновляем физику сразу после загрузки
    mujoco.mj_forward(model, data)
    return model, data

model, data = load_model()

with mujoco.viewer.launch(model, data) as viewer:
    print("Нажми F2 для перезагрузки модели, ESC для выхода")
    while viewer.is_running():
        if viewer.key_pressed == ord('F2'):
            print("Перезагружаем модель...")
            viewer.key_pressed = 0 # Сбрасываем флаг
            
            model, data = load_model()
            viewer.model = model
            viewer.data = data
            viewer.sync()
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005) # Задержка, чтобы точно поймать нажатие и не грузить CPU