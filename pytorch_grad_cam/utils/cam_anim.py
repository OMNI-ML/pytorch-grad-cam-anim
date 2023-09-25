import os
import ffmpeg
import subprocess
import matplotlib.pyplot as plt
from .image import show_cam_on_image


def count_parameters(model):
  """https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

  Args:
      model (_type_): _description_

  Returns:
      _type_: _description_
  """
  
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _ffmpeg_standard_quality(tmp_path, output_fname, frame_rate=5):
  """ _ffmpeg_standard_quality
      Generates and saves-to-file the animated .MP4 video in standard quality.
      ---
      Input: tmp_path <str>, the path to the images
             output_fname <str>, the path and filename for the saved file
             frame_rate=5 <int>, the number of frames per second
      Output: None
  """
  print('Generating video with pngs from:\n', tmp_path)
  try:
    # Define the command
    command = ['ffmpeg', '-framerate', str(frame_rate), '-pattern_type', 'glob', '-i', tmp_path + '/*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_fname]

    # Run the command and capture all output channels
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Print the output
    print(result.stdout)
    print(result.stderr)
  except Exception as exc:
    print('ERROR: ffmpeg video generation failed; video corrupt.')
    print(exc)
  return None


def _ffmpegpy_standard_quality(tmp_path, output_fname, frame_rate=5):
  """ _ffmpeg_standard_quality
      Generates and saves-to-file the animated .MP4 video in standard quality.
      ---
      Input: tmp_path <str>, the path to the images
             output_fname <str>, the path and filename for the saved file
             frame_rate=5 <int>, the number of frames per second
      Output: None
  """
  try:
    (
      ffmpeg
      .input(tmp_path + '*.png', pattern_type='glob', framerate=frame_rate)
      .output(output_fname)
      .run()
    )
  except Exception as exc:
    print('ERROR: ffmpeg video generation failed; video corrupt.')
    print(exc)
  return None


def _ffmpeg_high_quality(tmp_path, output_fname, frame_rate=5):
  """ _ffmpeg_standard_quality
      Generates and saves-to-file the animated .MP4 video in standard quality.
      ---
      Input: tmp_path <str>, the path to the images
             output_fname <str>, the path and filename for the saved file
             frame_rate=5 <int>, the number of frames per second
      Output: None
  """
  print('Generating video with pngs from:\n', tmp_path)
  try:
    # Define the command
    command = [
    'ffmpeg',
    '-framerate', str(frame_rate),
    '-pattern_type', 'glob',
    '-i', tmp_path + '/*.png',
    '-filter:v', 'scale=hd1080:force_original_aspect_ratio=increase',
    '-c:v', 'libx264',
    '-crf', '20',
    '-preset', 'slower',
    '-movflags', '+faststart',
    '-pix_fmt', 'yuv420p',
    output_fname
]

    # Run the command and capture all output channels
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Print the output
    print(result.stdout)
    print(result.stderr)
  except Exception as exc:
    print('ERROR: ffmpeg video generation failed; video corrupt.')
    print(exc)
  return None


def _ffmpegpy_high_quality(tmp_path, output_fname, frame_rate=5):
  """ _ffmpeg_high_quality
      Generates and saves-to-file the animated .MP4 video in high quality.
      Resulting video is scaled to 1080p.
      ---
      Input: tmp_path <str>, the path to the images
             output_fname <str>, the path and filename for the saved file
             frame_rate=5 <int>, the number of frames per second
      Output: None
  """
  try:
    (
      ffmpeg
      .input(tmp_path + '*.png', pattern_type='glob', framerate=frame_rate)
      .filter('scale', size='hd1080', force_original_aspect_ratio='increase')
      .output(output_fname, crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p')
      .run()
    )
  except Exception as exc: 
    print('ERROR: ffmpeg video generation failed; video corrupt.')
    print(exc)
    
  return None


def create_image_as_png(img, key, val, layer_name_map, out_dir):

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    if not out_dir.endswith(os.sep): out_dir += os.sep
  
    cam_image = show_cam_on_image(img, val[0, :], use_rgb=True)
    fig = plt.figure()
    plt.imshow(cam_image)
    plt.title(layer_name_map[key])
    fig.savefig(out_dir + key + ".png")
    plt.close(fig)

