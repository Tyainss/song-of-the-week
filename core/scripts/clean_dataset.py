
from common.logging import setup_logging
from common.config_manager import ConfigManager
from core.cleaning.pipeline import run_pipeline
from pathlib import Path

def main():
    cm = ConfigManager(Path.cwd())
    project_cfg = cm.project()
    setup_logging(project_cfg)
    run_pipeline()

if __name__ == "__main__":
    main()
