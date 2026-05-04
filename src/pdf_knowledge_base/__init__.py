from constants import PROJECT_ROOT


from loguru import logger

logger.add(
    PROJECT_ROOT / "logs/pdf-knowledge-base.log",
    rotation="1 day",
    retention="5 days",
)
logger.info("logger initialized")
