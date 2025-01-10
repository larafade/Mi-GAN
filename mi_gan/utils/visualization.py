"""
Author: Lara Fadel
Date: 2024-12-19
MCGill Composite Center
Department of Chemical Engineering and Material Science, University of Southern California
Email: larafade@usc.edu

Provides functions to visualize and save outputs.
"""

from PIL import Image, ImageDraw

def save_table_image(results, output_path):
    """
    Saves a table of input and output images side by side.

    Args:
        results (list of tuple): List of (input_image, output_image) tuples.
        output_path (str): Path to save the table image.
    """
    num_images = len(results)
    if num_images == 0:
        raise ValueError("No results to save.")

    image_width, image_height = results[0][0].size
    table_width = image_width * 2
    table_height = image_height * num_images

    table_image = Image.new('RGB', (table_width, table_height))

    for idx, (input_image, output_image) in enumerate(results):
        y_offset = idx * image_height
        table_image.paste(input_image, (0, y_offset))
        table_image.paste(output_image, (image_width, y_offset))

        draw = ImageDraw.Draw(table_image)
        draw.text((10, y_offset + 10), "Input", fill="red")
        draw.text((image_width + 10, y_offset + 10), "Output", fill="red")

    table_image.save(output_path)
    print(f"Saved table image at {output_path}")
