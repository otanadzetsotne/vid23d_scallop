import json
import subprocess

from moviepy.editor import VideoFileClip


def get_codec_info(video_file):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name",
        "-of", "json",
        video_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def extract_and_add_audio(
        source_video_path,
        target_video_path,
        output_video_path,
):
    # Load the source video
    source_video = VideoFileClip(source_video_path)
    source_codec = get_codec_info(source_video_path)
    # Extract audio from the source video
    audio_clip = source_video.audio

    # Load the target video (to which we want to add audio)
    target_video = VideoFileClip(target_video_path)
    # Add the extracted audio to the target video
    final_video = target_video.set_audio(audio_clip)

    # Write the result to a new file
    final_video.write_videofile(
        output_video_path,
        codec=source_codec['streams'][0]['codec_name'],
        audio_codec=source_codec['streams'][1]['codec_name'],
    )

    # Close the clips to free up system resources
    source_video.close()
    target_video.close()
    audio_clip.close()


if __name__ == '__main__':
    extract_and_add_audio(
        '/Users/otana/Development/vid23d/.results/main/main.mp4',
        '/Users/otana/Development/vid23d/.results/main/result/full.mp4',
        '/Users/otana/Development/vid23d/.results/main/result/full_audio.mp4'
    )
