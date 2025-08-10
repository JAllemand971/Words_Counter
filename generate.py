#!/usr/bin/env python3
"""
generate.py

This script processes a French video file to:
1. Transcribe its audio using Whisper and identify where specific words occur.
2. Compute precise timestamps for each occurrence based on each word's position.
3. Overlay a live word count onto the video with OpenCV, writing out a new video file.

Key steps:
- Load Whisper ASR model to convert audio into text segments.
- Normalize text (remove accents) and split into tokens (words).
- Map each token index to a time within its segment.
- Use MoviePy to iterate video frames, updating counters when frame time passes each word timestamp.
"""

import json
import re
import cv2
import unicodedata
import whisper
from moviepy.editor import VideoFileClip

# === Configuration ===
# Words we want to count in the transcript (lowercase, without accents)
TARGET_WORDS = ['genre', 'littéralement']
# Optional delay before incrementing display, to improve sync (seconds)
DISPLAY_DELAY_SECONDS = 0.0
# Colors for each counter text (BGR format)
COUNTER_COLORS = {
    'genre': (0, 0, 255),          # red for "genre"
    'littéralement': (0, 165, 255) # orange for "littéralement"
}
# Overlay text positions
TEXT_START_Y = 50      # Y position for the first line of text
TEXT_LINE_SPACING = 40 # Vertical spacing between lines


def remove_accents(input_text: str) -> str:
    """
    Remove accent marks from input text.
    Converts characters like 'é' to 'e', 'à' to 'a'.
    """
    normalized_text = unicodedata.normalize('NFD', input_text)
    return ''.join(
        char for char in normalized_text
        if unicodedata.category(char) != 'Mn'
    )


def transcribe_and_timestamp(video_path: str, words_to_find: list[str]) -> dict[str, list[float]]:
    """
    Transcribe the video, then determine timestamps for each target word.

    1. Use Whisper to transcribe audio into segments with start/end times.
    2. For each segment:
       a. Remove accents and lowercase the text.
       b. Split text into word tokens, including apostrophes.
       c. For each target word, find token positions matching it.
       d. Convert each token index into an exact timestamp within the segment.

    Returns:
        Dict mapping each target word to a sorted list of timestamps (in seconds).
    """
    # Load ASR model
    whisper_model = whisper.load_model('large')
    # Perform transcription; result contains 'segments'
    transcription_result = whisper_model.transcribe(video_path, language='fr')

    # Prepare data structure for timestamps
    word_timestamps: dict[str, list[float]] = {
        word: [] for word in words_to_find
    }
    # Pre-normalize each target word for matching
    normalized_targets = {
        word: remove_accents(word).lower()
        for word in words_to_find
    }

    # Process each segment returned by Whisper
    for segment in transcription_result['segments']:
        segment_start = segment['start']  # start time of this segment (sec)
        segment_end = segment['end']      # end time of this segment (sec)
        segment_duration = segment_end - segment_start

        # Normalize segment text and split into tokens
        raw_text = segment['text']
        normalized_text = remove_accents(raw_text).lower()
        # Regex matches words with optional apostrophes
        token_list = re.findall(r"\b\w+['’]?\w*\b", normalized_text)
        number_of_tokens = len(token_list) or 1

        # For each target word, locate matching tokens
        for original_word, normalized_word in normalized_targets.items():
            # Find all token indices equal to the target
            matching_indices = [
                idx for idx, token in enumerate(token_list)
                if token == normalized_word
            ]
            # Convert each match index to a timestamp
            for token_index in matching_indices:
                # place timestamp at the midpoint of this token's time span
                relative_position = (token_index + 0.5) / number_of_tokens
                word_time = (
                    segment_start + relative_position * segment_duration
                )
                word_timestamps[original_word].append(word_time)

    # Sort timestamps and print for debugging
    for word, timestamps in word_timestamps.items():
        timestamps.sort()
        print(f"Detected {len(timestamps)} '{word}' at: {timestamps}")

    return word_timestamps


def overlay_word_counters(
    video_path: str,
    word_timestamps: dict[str, list[float]],
    output_path: str
) -> None:
    """
    Create a new video with live counters overlaid.

    1. Load the original video via MoviePy.
    2. For each frame at time t:
       - Check each word's timestamp list: if t >= timestamp + delay, increment its counter.
       - Draw updated counter text on the frame using OpenCV.
    3. Write out the annotated video, preserving audio.
    """
    # Load video file
    video_clip = VideoFileClip(video_path)
    # Initialize counters and index pointers for each word
    word_counts = {word: 0 for word in word_timestamps}
    index_pointers = {word: 0 for word in word_timestamps}

    def draw_counters(get_frame_function, current_time: float):
        """Callback to draw counters on each video frame."""
        # Get raw RGB frame data
        rgb_frame = get_frame_function(current_time)
        # Convert to BGR for OpenCV operations
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Update and draw each word's counter
        for line_index, (word, timestamps) in enumerate(word_timestamps.items()):
            # Increment while current time passes the next timestamp
            while (
                index_pointers[word] < len(timestamps)
                and current_time >= timestamps[index_pointers[word]] + DISPLAY_DELAY_SECONDS
            ):
                word_counts[word] += 1
                index_pointers[word] += 1

            # Prepare display text and position
            display_label = remove_accents(word).upper()
            display_text = f"{display_label} said: {word_counts[word]} times"
            text_position = (
                10,  # X: 10 pixels from left
                TEXT_START_Y + line_index * TEXT_LINE_SPACING
            )
            # Draw text with anti-aliasing
            cv2.putText(
                bgr_frame,
                display_text,
                text_position,
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                COUNTER_COLORS[word],
                2,
                lineType=cv2.LINE_AA
            )

        # Convert back to RGB for MoviePy
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Apply the annotation function to video frames
    annotated_clip = video_clip.fl(
        lambda gf, t: draw_counters(gf, t),
        apply_to=['video']
    )
    # Write out new video file with audio
    annotated_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )


def main() -> None:
    """Main entry: specify paths, run transcription+timestamping and overlay."""
    input_video_path = 'RiriTest.mp4'
    timestamps_output_path = 'times.json'
    final_video_output_path = 'video_with_counters.mp4'

    # 1. Transcribe and compute word timestamps
    timestamps = transcribe_and_timestamp(
        input_video_path, TARGET_WORDS
    )
    # Save timestamps for inspection (optional)
    with open(timestamps_output_path, 'w', encoding='utf-8') as timestamp_file:
        json.dump(timestamps, timestamp_file)

    # 2. Overlay counters onto the video
    overlay_word_counters(
        input_video_path,
        timestamps,
        final_video_output_path
    )


if __name__ == '__main__':
    main()
