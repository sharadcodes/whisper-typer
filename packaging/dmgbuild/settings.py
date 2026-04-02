import os

app_name = os.environ.get("APP_NAME", "Whisper Typer")
app_path = os.environ["APP_PATH"]  # full path to the .app
out_dmg = os.environ.get("DMG_NAME", "Whisper-Typer.dmg")

volume_name = app_name
filename = out_dmg

icon = None
badge_icon = None

background = None
window_rect = ((200, 200), (640, 420))
default_view = "icon-view"
icon_size = 128

files = [app_path]
symlinks = {"Applications": "/Applications"}

# Reasonable default layout.
icon_locations = {
    os.path.basename(app_path): (180, 200),
    "Applications": (460, 200),
}

