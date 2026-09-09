# sim/view_scene.py
import mujoco
import mediapy as media
import os

# Указываем бэкенд рендеринга (если нужно)
os.environ["MUJOCO_GL"] = "glfw"  # или "egl", если glfw не работает

# Загружаем сцену
model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)

# Делаем один шаг, чтобы обновить состояние
mujoco.mj_forward(model, data)

# Создаём рендерер для overhead-камеры
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="overhead")
pixels = renderer.render()

# Показываем картинку (в Jupyter или сохраняем)
# Вариант 1: если ты в Jupyter, используй media.show_image
# media.show_image(pixels)

# Вариант 2: сохраняем в файл, чтобы открыть в любом просмотрщике
media.write_image("sim/overhead_view.png", pixels)
print("Сохранено: sim/overhead_view.png")

# Дополнительно: если хочешь увидеть и другие камеры
for cam in ["wrist_left", "wrist_right"]:
    renderer.update_scene(data, camera=cam)
    pixels_cam = renderer.render()
    media.write_image(f"sim/{cam}_view.png", pixels_cam)
    print(f"Сохранено: sim/{cam}_view.png")