import argparse

from stylizer import Stylizer


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--reference-path", type=str, required=True, help="reference .obj path")
  parser.add_argument("-i", "--input-path", type=str, required=True, help="input .obj path")
  parser.add_argument("-o", "--output-path", type=str, required=True, help="output .obj path")
  parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    required=False,
    default=1,
    help="tuning parameter for intensity of stylization",
  )
  args = parser.parse_args()

  stylizer = Stylizer()
  stylizer.run(args.reference_path, args.input_path, args.output_path, max(0, args.gamma))
