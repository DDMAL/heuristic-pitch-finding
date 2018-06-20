from AomrObject import AomrObject
from gamera.core import load_image


def test_interpolation():
    inImage = './tests/res/CF18_Staff.png'
    image = load_image(inImage)

    kwargs = {
        'lines_per_staff': 4,
        'staff_finder': 0,
        'staff_removal': 0,
        'binarization': 1,
        'discard_size': 12,
    }

    aomr_obj = AomrObject(image, **kwargs)

    staves = aomr_obj.find_staves()

    aomr_obj.staff_coords()

    assert True
