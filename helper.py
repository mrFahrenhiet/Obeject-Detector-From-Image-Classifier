import imutils


def pyramid(image, scale=1.5, min_size=(224, 224)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, ws):
    for y in range(0, image.shape[0] - ws[1], step_size):
        for x in range(0, image.shape[1] - ws[0], step_size):
            yield (x, y, image[y: y + ws[1], x: x + ws[0]])


