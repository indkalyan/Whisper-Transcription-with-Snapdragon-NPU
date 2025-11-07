import logging
from FileHelper import FileHelper
from pathlib import Path

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, USLT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_lyrics(dirOrFilePath):
    lyrics_found = False
    audio_files = FileHelper.getFilesAsList(dirOrFilePath,"*.mp3")
    for audio_file in audio_files:
        audio_name = Path(audio_file).stem
        output_file = dirOrFilePath / f"{audio_name}_transcript.txt"
        audio = MP3(audio_file, ID3=ID3)
        with open(output_file, "w", encoding="utf-8") as f:
            for tag in audio.tags.values():
                if isinstance(tag, USLT):
                    f.write(tag.text.strip())
                    lyrics_found = True
                    break
    if lyrics_found:
        print(f"Lyrics successfully extracted to: {output_file}")
    else:
        print("No lyrics (USLT tag) found in this MP3 file.")





def main():
    extract_lyrics(fileDirPath = Path("C:\\Users\\indka\\Music\\pons"))

    # Example usage
if __name__ == "__main__":
    main()