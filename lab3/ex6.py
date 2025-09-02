import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def sobel_edge_detection(img):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)

    G = np.hypot(Gx, Gy)
    G = np.clip(G, 0, 255)
    return G.astype(np.uint8)


def main():
    # Load the image
    img = Image.open("test.png").convert('L')
    img = np.array(img)

    # Apply Sobel edge detection
    sobel_img = sobel_edge_detection(img)

    # Convert the result to an image and save it
    sobel_img = Image.fromarray(sobel_img)
    sobel_img.convert('RGB').save("output.jpg")

if __name__ == "__main__":
    main()