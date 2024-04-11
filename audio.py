from moviepy.editor import VideoFileClip


def extract_and_add_audio(
        source_video_path,
        target_video_path,
        output_video_path,
):
    # Load the source video
    source_video = VideoFileClip(source_video_path)
    # Extract audio from the source video
    audio_clip = source_video.audio

    # Load the target video (to which we want to add audio)
    target_video = VideoFileClip(target_video_path)
    # Add the extracted audio to the target video
    final_video = target_video.set_audio(audio_clip)

    # Write the result to a new file
    final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # Close the clips to free up system resources
    source_video.close()
    target_video.close()
    audio_clip.close()
