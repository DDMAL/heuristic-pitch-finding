import sys
from gamera.core import load_image, init_gamera, ONEBIT
from gamera import gamera_xml

from StaffProcessing import StaffProcessor
from PitchFinding import PitchFinder

import json

init_gamera()


def _binarize_image(image):
    image_resolution = image.resolution
    image = image.to_greyscale()
    bintypes = ['threshold',
                'otsu_threshold',
                'sauvola_threshold',
                'niblack_threshold',
                'gatos_threshold',
                'abutaleb_threshold',
                'tsai_moment_preserving_threshold',
                'white_rohrer_threshold']
    image = getattr(image, bintypes[1])(0)
    # BUGFIX: sometimes an image loses its resolution after being binarized.
    if image.resolution < 1:
        image.resolution = image_resolution

    # check the amount of blackness of the image. If it's inverted,
    # the black area will vastly outweigh the white area.
    area = image.area().tolist()[0]
    black_area = image.black_area()[0]

    if area == 0:
        raise AomrError("Cannot divide by a zero area. Something is wrong.")

    # if greater than 70% black, invert the image.
    if (black_area / area) > 0.7:
        image.invert()

    return image


if __name__ == "__main__":
    # given CC and a staff image, will generate a complete JSOMR file

    (tmp, inCC, inImage) = sys.argv

    image = load_image(inImage)
    glyphs = gamera_xml.glyphs_from_xml(inCC)
    kwargs = {
        'scanlines': 20,
        'lines_per_staff': 4,
        'tolerance': -1,
        'blackness': 0.8,
        'interpolation': True,

        'staff_finder': 0,
        'binarization': 1,

        'discard_size': 12,
    }

    sf = StaffProcessor(**kwargs)
    pf = PitchFinder(**kwargs)

    # do some binarization/dilation boi
    if image.data.pixel_type != ONEBIT:
        image = _binarize_image(image)
    image = image.dilate()
    image = image.dilate()
    image = image.dilate()

    page = sf.get_page_properties(image)
    polys = sf.get_staves(image)
    staves = sf.process_staves(polys)

    pitches = pf.get_pitches(glyphs, staves)

    jsomr = {
        'page': page,
        'staves': staves,
        'glyphs': pitches,
    }

    with open('jsomr_output.json', 'w') as f:
        f.write(json.dumps(jsomr))
        # print jsomr
