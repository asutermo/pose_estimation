import logging

__all__ = ["config_logs"]


def config_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )