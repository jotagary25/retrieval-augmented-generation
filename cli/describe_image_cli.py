import argparse
import mimetypes

from test_gemini import generate_multimodal


def main():
    parser = argparse.ArgumentParser(description="Describe an image.")

    parser.add_argument("--image", type=str, help="Path to the image.")
    parser.add_argument("--query", type=str, help="Query to describe the image.")

    args = parser.parse_args()
    image = args.image
    query = args.query

    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    with open(image, "rb") as f:
        image_data = f.read()

    response, tokens = generate_multimodal(image_data, mime, query)
    print(f"Rewritten query: {response}")
    print(f"Total tokens: {tokens}")


if __name__ == "__main__":
    main()
