"""
Script to add figure labels to visualization images.
"""
import os
import re
from PIL import Image, ImageDraw, ImageFont
import pathlib

# Configuration variables
SOURCE_FOLDER = "backend/simulation_results/main_analysis/analysis_20250422_082939/figures"  # Main analysis figures
DEFAULT_FONT_SIZE = 60  # Default font size

# Labels with configurable font sizes per image - organized by logical groups
LABELS = {
    # Algorithm-specific matchup analysis
    "matchup_mcts_lightweight_4_1.0.png": {
        "text": "Figure 32: Performance of Top MCTS Configuration (Lightweight Policy, Depth 4, Cp=1.0)",
        "font_size": 75
    },
    "matchup_mcts_lightweight_6_1.0.png": {
        "text": "Figure 33: Performance of Second MCTS Configuration (Lightweight Policy, Depth 6, Cp=1.0)",
        "font_size": 75
    },
    "matchup_mcts_lightweight_6_1.414.png": {
        "text": "Figure 34: Performance of Third MCTS Configuration (Lightweight Policy, Depth 6, Cp=√2)",
        "font_size": 75
    },
    
    # Game dynamics analysis
    "algorithm_draw_tendencies.png": {
        "text": "Figure 35: Draw Tendencies by Algorithm",
        "font_size": 75
    },
    "captures_vs_length.png": {
        "text": "Figure 36: Relationship Between Game Length and Number of Captures",
        "font_size": 75
    },
    "first_capture_timing.png": {
        "text": "Figure 37: Timing of First Goat Capture by Algorithm",
        "font_size": 75
    },
    
    # Opening move analysis
    "response_Goat_at_(0,2).png": {
        "text": "Figure 38: Algorithm Responses to Goat Placement at (0,2)",
        "font_size": 75
    },
    "response_Goat_at_(2,0).png": {
        "text": "Figure 39: Algorithm Responses to Goat Placement at (2,0)",
        "font_size": 75
    },
    "response_Goat_at_(2,4).png": {
        "text": "Figure 40: Algorithm Responses to Goat Placement at (2,4)",
        "font_size": 75
    },
    "response_Goat_at_(4,2).png": {
        "text": "Figure 41: Algorithm Responses to Goat Placement at (4,2)",
        "font_size": 75
    }
}

# Other settings
FONT_COLOR = (0, 0, 0)  # Black
PADDING = 30  # Padding below the image for the label

def extract_figure_number(label_text):
    """
    Extract the figure number from the label text.
    
    Args:
        label_text: The label text containing "Figure X: ..."
    
    Returns:
        The figure number as a string, or an empty string if not found
    """
    match = re.search(r'Figure (\d+)', label_text)
    if match:
        return match.group(1)
    return ""

def add_label_to_image(image_path, output_path, label_info):
    """
    Add a label to the bottom of an image.
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the labeled image
        label_info: Dictionary with label text and font size
    """
    # Open the image, preserving mode (RGB, RGBA, etc.)
    with Image.open(image_path) as img:
        original_mode = img.mode
        
        # Create a new image with extended height for the label
        width, height = img.size
        font_size = label_info.get("font_size", DEFAULT_FONT_SIZE)
        new_height = height + PADDING + font_size + PADDING
        
        # Create new image with white background regardless of transparency
        new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
        
        # Paste the original image at the top, preserving transparency if needed
        if original_mode == 'RGBA':
            # Create a solid white background first
            bg = Image.new('RGB', (width, height), (255, 255, 255))
            # Paste the original image onto the white background using alpha compositing
            bg.paste(img, (0, 0), img)
            # Then paste this composite onto the new image
            new_img.paste(bg, (0, 0))
        else:
            # For non-transparent images, just paste directly
            new_img.paste(img, (0, 0))
        
        # Add the label text
        draw = ImageDraw.Draw(new_img)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            # On macOS, try system fonts if Arial isn't found
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except IOError:
                font = ImageFont.load_default()
        
        # Calculate text position to center it
        label_text = label_info.get("text", "")
        text_width = draw.textlength(label_text, font=font)
        text_position = ((width - text_width) // 2, height + PADDING)
        
        # Draw the text
        draw.text(text_position, label_text, font=font, fill=FONT_COLOR)
        
        # Save the new image, preserving format
        new_img.save(output_path)

def main():
    """Main function to process all images in the source folder."""
    # Create the output directory if it doesn't exist
    source_path = pathlib.Path(SOURCE_FOLDER)
    output_dir = source_path / "Labels"
    output_dir.mkdir(exist_ok=True)
    
    # Process each image in the source folder
    for filename in os.listdir(source_path):
        if filename in LABELS:
            # This is an image that needs a label
            image_path = source_path / filename
            
            # Extract figure number from label text
            label_text = LABELS[filename]["text"]
            figure_number = extract_figure_number(label_text)
            
            # Create output filename with figure number
            stem = pathlib.Path(filename).stem
            suffix = pathlib.Path(filename).suffix
            if figure_number:
                output_filename = f"{stem}_fig{figure_number}{suffix}"
            else:
                output_filename = f"{stem}_label{suffix}"
                
            output_path = output_dir / output_filename
            
            print(f"Adding label to {filename} as {output_filename}...")
            add_label_to_image(image_path, output_path, LABELS[filename])
    
    print(f"Labeled images saved to {output_dir}")

if __name__ == "__main__":
    main() 