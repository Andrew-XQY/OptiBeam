from PIL import Image, ImageDraw, ImageFont

def create_counting_gif(filename, frame_duration, start=0, end=999):
    frames = []
    
    # Set a larger image size to enhance clarity with the default font
    image_size = (300, 150)
    
    # Use the default font provided by PIL
    font = ImageFont.load_default()

    for i in range(start, end + 1):
        image = Image.new('RGB', image_size, color=(255, 255, 255))
        d = ImageDraw.Draw(image)
        text = str(i)
        # Calculate text width and height using textbbox
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        textwidth = right - left
        textheight = bottom - top
        x = (image.width - textwidth) / 2
        y = (image.height - textheight) / 2
        d.text((x, y), text, fill=(0, 0, 0), font=font)
        frames.append(image)
    
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_duration,
        loop=0
    )

create_counting_gif('../../ResultsCenter/counting_test_10ms.gif', frame_duration=10)







