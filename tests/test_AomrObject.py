import unittest
from AomrObject import AomrObject
from gamera.core import load_image


class T(unittest.TestCase):

    # def __init__(self):
    inImage = './tests/res/CF18_Staff.png'
    inGameraXML = './tests/res/CF18_Classified.xml'

    image = load_image(inImage)
    kwargs = {
        'lines_per_staff': 4,
        'staff_finder': 0,
        'staff_removal': 0,
        'binarization': 1,
        'discard_size': 12,
    }

    aomr_obj = None

    # unit tests
    def test_a01_generate_aomr_obj(self):
        T.aomr_obj = AomrObject(self.image, **self.kwargs)
        assert True

    #################
    # Staff Finding
    #################

    def test_b01_find_staves(self):
        T.aomr_obj.find_staves()
        T.staves, T.interpolated_staves = T.aomr_obj.get_staves()
        assert True

    def test_b02_sequential_points(self):
        # all points in a line are ordered by x position from left to right
        for i, staff in enumerate(T.staves):
            for j, line in enumerate(staff['line_positions']):
                for k, pt in enumerate(line[1:]):
                    if not pt[0] > line[k][0]:
                        print(pt, line[k])
                        assert False

        assert True

    #######################
    # Interpolation Tests
    #######################

    def test_c01_interpolation_grabbed_previous_points(self):
        # check all original points
        for i, staff in enumerate(T.staves):
            for j, line in enumerate(staff['line_positions']):
                for pt in line:

                    # all original points remain in interpolated line
                    if pt not in T.interpolated_staves[i]['line_positions'][j]:
                        print(pt)
                        assert False
        assert True

    def test_c02_interpolated_all_points(self):
        # check all interpolated points
        for i, staff in enumerate(T.interpolated_staves):
            for j, line in enumerate(staff['line_positions']):
                for pt in line:

                    # no missing points
                    if not pt[0] or not pt[1]:
                        assert False

        assert True

    def test_c03_find_right(self):
        tests = [None] * 4

        tests[0] = T.aomr_obj._find_right_pt([(1, 1), (False, False), (2, 2), (3, 3), (4, 4)], 0)
        tests[1] = T.aomr_obj._find_right_pt([(1, 1), (False, False), (False, False), (3, 3), (4, 4)], 0)
        tests[2] = T.aomr_obj._find_right_pt([(1, 1), (False, False), (False, False), (False, False), (4, 4)], 0)
        tests[3] = T.aomr_obj._find_right_pt([(1, 1), (False, False), (False, False), (False, False), (False, False)], 0)

        assert tests == [2, 3, 4, False]

    def test_c04_interpolated_ys_within_edges(self):
        for i, staff in enumerate(T.interpolated_staves):
            for j, line in enumerate(staff['line_positions']):
                original_line = T.staves[i]['line_positions'][j]

                for k, pt in enumerate(line[1:-1]):
                    if pt not in original_line:  # interpolated points should never be a max or min

                        slope_down = (pt[1] <= line[k - 1][1] and pt[1] >= line[k + 1][1])
                        slope_up = (pt[1] >= line[k - 1][1] and pt[1] <= line[k + 1][1])

                        if not (slope_down or slope_up):
                            assert False

        assert True
