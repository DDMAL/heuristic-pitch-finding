from gamera.core import *
from gamera.toolkits import musicstaves
# from gamera.toolkits.aomr_tk.AomrExceptions import *
from gamera import classify, knn, gamera_xml

import sys
import zipfile
import os
import warnings
import tempfile
import copy
import itertools
import random
import math
import json

from operator import itemgetter, attrgetter

import logging
lg = logging.getLogger('aomr')
import pdb

init_gamera()

#
#
# FOR TEST
from StaffFinding import StaffFinder
from PitchFinding import PitchFinder
#
#
#


class AomrObject(object):
    """
    Manipulates an Aomr file and stores its information
    """

    def __init__(self, image, **kwargs):
        """
            Constructs and returns an AOMR object
        """

        # binarization
        if self.image.data.pixel_type != ONEBIT:
            self.image = self.image.to_greyscale()
            bintypes = ['threshold',
                        'otsu_threshold',
                        'sauvola_threshold',
                        'niblack_threshold',
                        'gatos_threshold',
                        'abutaleb_threshold',
                        'tsai_moment_preserving_threshold',
                        'white_rohrer_threshold']
            self.image = getattr(self.image, bintypes[self.binarization])(0)
            # BUGFIX: sometimes an image loses its resolution after being binarized.
            if self.image.resolution < 1:
                self.image.resolution = self.image_resolution

        # check the amount of blackness of the image. If it's inverted,
        # the black area will vastly outweigh the white area.
        area = self.image.area().tolist()[0]
        black_area = self.image.black_area()[0]

        if area == 0:
            raise AomrError("Cannot divide by a zero area. Something is wrong.")

        # if greater than 70% black, invert the image.
        if (black_area / area) > 0.7:
            self.image.invert()

        self.image_size = [self.image.ncols, self.image.nrows]

        self.page_result = {
            'staves': {},
            'dimensions': self.image_size
        }

    ####################
    # Public Functions
    ####################

    def get_page_properties(self):
        return {
            # 'filename': self.image,
            'resolution': self.image_resolution,
            'bounding_box': {
                'ncols': self.image.ncols,
                'nrows': self.image.nrows,
                'ulx': 0,
                'uly': 0,
            }
        }

    def get_staves(self):
        return [self.staves, self.interpolated_staves]

    def get_staves_properties(self):
        num_staves = len(self.interpolated_staves)

        staves = []
        for st in self.interpolated_staves:
            staff = {
                'coords': st['coords'],
                'num_lines': st['num_lines'],
                'staff_no': st['staff_no'],
            }

            staves.append(staff)

        return {
            'num_staves': num_staves,
            'staves': staves,
        }

    #####################
    # Private Functions
    #####################

    def _px_to_mm10(self, pixels):
        # 25.4 mm in an inch * 10
        return int(round((pixels * 254) / self.image.resolution))

    def _get_staff_by_coordinates(self, x, y):
        for k, v in self.page_result['staves'].iteritems():
            top_coord = v['line_positions'][0][0]
            bot_coord = v['line_positions'][-1][-1]

            # y is the most important for finding which staff it's on
            if top_coord[1] <= y <= bot_coord[1]:
                # add 20 mm10 to the x values, since musicstaves doesn't
                # seem to accurately guess the starts and ends of staves.
                if top_coord[0] - self._px_to_mm10(20) <= x <= bot_coord[0] + self._px_to_mm10(20):
                    return k
        return None


if __name__ == "__main__":
    (tmp, inCC, inImage) = sys.argv

    # open files to be read
    # fImage = open(inImage, 'r')

    image = load_image(inImage)
    glyphs = gamera_xml.glyphs_from_xml(inCC)
    kwargs = {
        'lines_per_staff': 4,
        'staff_finder': 0,
        'binarization': 1,
        'interpolation': True,
    }

    jsomr = {
        'page': [],
        'staves': [],
        'glyphs': [],
    }

    sf = StaffFinder(**kwargs)
    staves = sf.get_staves(image)

    print staves
    pf = PitchFinder()
    pitches = pf.get_pitches(glyphs, staves)

    # print(staves)

    # aomr_obj = AomrObject(image, **kwargs)
    # staves = aomr_obj.find_staves()                # returns true!
    # # print aomr_obj.get_staves_properties()
    # # print staves
    # sorted_glyphs = aomr_obj.miyao_pitch_finder(glyphs)  # returns what we want

    # pitch_feature_names = ['staff', 'offset', 'strt_pos', 'note', 'octave', 'clef_pos', 'clef']

    # # get page information
    # output_json['page'] = aomr_obj.get_page_properties()

    # # get glyphs information
    # for i, g in enumerate(sorted_glyphs):

    #     cur_json = {}
    #     pitch_info = {}
    #     glyph_info = {}

    #     # get pitch information
    #     for j, pf in enumerate(g[1:]):
    #         pitch_info[pitch_feature_names[j]] = str(pf)
    #     cur_json['pitch'] = pitch_info

    #     # get glyph information
    #     glyph_info['bounding_box'] = {
    #         'ncols': g[0].ncols,
    #         'nrows': g[0].nrows,
    #         'ulx': g[0].ul.x,
    #         'uly': g[0].ul.y,
    #     }
    #     glyph_info['state'] = gamera_xml.classification_state_to_name(g[0].classification_state)
    #     glyph_info['name'] = g[0].id_name[0][1]
    #     cur_json['glyph'] = glyph_info

    #     output_json['glyphs'].append(cur_json)

    # with open('jsomr_output.json', 'w') as f:
    #     f.write(json.dumps(output_json))
    #     # print output_json
