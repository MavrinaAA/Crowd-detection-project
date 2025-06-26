import argparse
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="CrowdYOLO - интуитивный запуск")
    parser.add_argument('--model', type=str, required=True, help='Путь к модели YOLO (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Путь к входному видео')
    parser.add_argument('--output', type=str, default='output/crowd_annotated.mp4', help='Путь для выходного результата')

    args = parser.parse_args()

    run_inference(args.model, args.source, args.output)

if __name__ == "__main__":
    main()