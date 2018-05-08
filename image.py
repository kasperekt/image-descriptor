import numpy as np


def get_patch(image, keypoint, size=(32, 32)):
    ''' Z treści zadania:
    "Żaden punkt nie będzie bliżej brzegu obrazu niż 32 piksele."
    "Funkcja do wyliczenie pojedynczego deskryptora nie powinna używać okna
    większego niż 64×64 wycentrowanego na interesującym nas punkcie."

    Dlatego nie ma sprawdzania czy obszar "patcha" wychodzi poza obraz.
    '''
    y_size, x_size = size
    x, y = keypoint

    left_bound = int(x - x_size // 2)
    right_bound = int(x + x_size // 2)
    top_bound = int(y - y_size // 2)
    bottom_bound = int(y + y_size // 2)

    return image[top_bound:bottom_bound,
                 left_bound:right_bound]
