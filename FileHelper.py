import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileHelper:


    @staticmethod
    def getFilesAsList(dirOrFilePath, filetype) -> list[str]:
        if dirOrFilePath.is_file():
            files = [str(dirOrFilePath)]
        elif dirOrFilePath.is_dir():
            files = list(dirOrFilePath.glob(filetype))
            files = [str(f) for f in files]
            if not files:
                logger.error(f"❌ No MP3 files found in {dirOrFilePath}")
                sys.exit(1)
        else:
            logger.error(f"❌ Invalid path: {dirOrFilePath}")
            sys.exit(1)
        logger.info(f"Found {len(files)} audio file(s)")
        return files
