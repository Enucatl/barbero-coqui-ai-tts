import collections
import logging

import click
import webrtcvad
import pydub


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator) from pydub.AudioSegment
    Returns: A generator that yields PCM audio data.
    """
    logger = logging.getLogger(__name__)
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = pydub.AudioSegment.empty()
    for frame in frames:
        assert frame.frame_rate == sample_rate
        assert frame.channels == 1
        # drop frame if too short, which can happen at the end of the file
        if len(frame) < frame_duration_ms:
            continue
        is_speech = vad.is_speech(frame.raw_data, sample_rate)
        logger.debug("1" if is_speech else "0")
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                logger.debug("+(%s) frames", voiced_frames.frame_count())
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, _ in ring_buffer:
                    voiced_frames = voiced_frames.append(f, crossfade=0)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames = voiced_frames.append(frame, crossfade=0)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                logger.debug("-(%s) frames", voiced_frames.frame_count())
                triggered = False
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = pydub.AudioSegment.empty()
    # If we have any leftover voiced audio when we run out of input, yield it.
    if triggered:
        logger.debug("-(%s) frames", frame.frame_count())
    if voiced_frames.frame_count() > 0:
        yield voiced_frames


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def main(input_file):
    logging.basicConfig(level=logging.DEBUG)
    sound = pydub.AudioSegment.from_file(input_file)
    logging.debug("sound: %s", sound[::10])
    collection = vad_collector(16000, 10, 110, webrtcvad.Vad(3), sound[::10])
    for i, segment in enumerate(collection):
        print(i, segment.frame_count())

if __name__ == "__main__":
    main()
