CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

def rgb_to_ansi(r, g, b):
    GAMMA = 1
    r, g, b = int(r ** GAMMA * 255), int(g ** GAMMA * 255), int(b ** GAMMA * 255)
    r, g, b = r // 43, g // 43, b // 43
    return 16 + (36 * r) + (6 * g) + b

def add_border(image, border_size=1, border_color=(0.5, 0.5, 0.5)):
    new_image = [[[border_color[channel]] * (32 + 2 * border_size) for channel in range(3)] for _ in range(3)]
    for c in range(3):
        for h in range(32):
            for w in range(32):
                new_image[c][h + border_size][w + border_size] = image[c][h][w]
    return new_image

def normalize_brightness(image, scale=1.0):
    max_pixel = max(max(max(row) for row in channel) for channel in image)
    if max_pixel == 0:
        return image
    return [[[pixel * scale / max_pixel for pixel in row] for row in channel] for channel in image]

def to_greyscale(image):
    return [[sum(image[channel][h][w] for channel in range(3)) / 3 for w in range(32)] for h in range(32)]

def image_to_ascii(image):
    greyscale_map = " .:-=+*#%@"
    ascii_image = ""
    for h in range(len(image)):
        for w in range(len(image[h])):
            pixel_value = image[h][w]
            greyscale_index = int(pixel_value * (len(greyscale_map) - 1))
            ascii_image += greyscale_map[greyscale_index]
        ascii_image += "\n"
    return ascii_image

def display_image_with_prediction(image, true_label, predicted_class, confidence, index, show_greyscale=False, show_ascii=False):
    true_label_name = CIFAR10_CLASSES[true_label]
    predicted_label_name = CIFAR10_CLASSES[predicted_class]
    correct = (true_label == predicted_class)
    
    true_label_display = f"\033[92mExpected Class: {true_label_name:<15}\033[0m" if correct else f"\033[91mExpected Class: {true_label_name:<15}\033[0m"
    predicted_label_display = f"\033[92mPredicted Class: {predicted_label_name:<15}\033[0m" if correct else f"\033[91mPredicted Class: {predicted_label_name:<15}\033[0m"

    print(f" Sample {index + 1} ".center(60, "="))
    print(f"| {true_label_display}".ljust(59) + "|")
    print(f"| {predicted_label_display}".ljust(59) + "|")
    print(f"| Confidence: {confidence:.2%}".ljust(59) + "|")
    print("=" * 60)
    print("+" + "-" * 64 + "+")

    if show_greyscale:
        image = to_greyscale(image)
    else:
        image = normalize_brightness(image)

    for h in range(32):
        print("|", end="")
        for w in range(32):
            if show_greyscale:
                intensity = image[h][w]
                color_code = rgb_to_ansi(intensity, intensity, intensity)
            else:
                r, g, b = image[0][h][w], image[1][h][w], image[2][h][w]
                color_code = rgb_to_ansi(r, g, b)
            print(f"\033[48;5;{color_code}m  ", end="")
        print("\033[0m|")
    print("+" + "-" * 64 + "+")

    if show_ascii:
        ascii_image = image_to_ascii(to_greyscale(image) if not show_greyscale else image)
        print(ascii_image)
    print("=" * 60 + "\n")