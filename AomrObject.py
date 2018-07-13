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


class AomrObject(object):
    """
    Manipulates an Aomr file and stores its information
    """

    def __init__(self, image, **kwargs):
        """
            Constructs and returns an AOMR object
        """
        self.SCALE = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        self.clef = 'c', 3              # default clef

        self.transpose = 0              # shift all notes by x 2nds
        self.space_proportion = 0.5     # space when within middle portion between lines
        self.get_staff_margin = 2.0     # glyphs are on staff when within x punctums above/below it
        self.always_find_staff_no = False

        # staff finding kwargs
        self.lines_per_staff = kwargs['lines_per_staff']
        self.sfnd_algorithm = kwargs['staff_finder']
        self.srmv_algorithm = kwargs['staff_removal']
        self.binarization = kwargs["binarization"]

        self.discard_size = kwargs["discard_size"]
        self.avg_punctum = None

        self.staffless_glyphs = ['skip']        # any staffless glyph IS pitchless
        self.pitchless_glyphs = ['division']

        # the result of the staff finder. Mostly for convenience
        self.staff_finder = None
        self.staves = None
        self.interpolated_staves = None
        self.staff_bounds = None

        # cache this once so we don't have to constantly load it
        self.image = image
        self.image_resolution = self.image.resolution

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

        if (black_area / area) > 0.7:
            # if it is greater than 70% black, we'll invert the image. This is an in-place operation.
            lg.debug("Inverting the colours.")
            self.image.invert()

        self.image_size = [self.image.ncols, self.image.nrows]

        # store the image without stafflines
        self.img_no_st = None
        self.rgb = None
        self.page_result = {
            'staves': {},
            'dimensions': self.image_size
        }

    ####################
    # Public Functions
    ####################

    def find_staves(self):
        if self.sfnd_algorithm is 0:
            s = musicstaves.StaffFinder_miyao(self.image)
        elif self.sfnd_algorithm is 1:
            s = musicstaves.StaffFinder_dalitz(self.image)
        elif self.sfnd_algorithm is 2:
            s = musicstaves.StaffFinder_projections(self.image)
        else:
            raise AomrStaffFinderNotFoundError("The staff finding algorithm was not found.")

        staves = self._find_staves(s)
        return self._process_staves(staves)

    def miyao_pitch_finder(self, glyphs):
        """
            Returns a set of glyphs with pitches
        """
        glyphs = list(filter(lambda g: g.get_main_id() != 'skip', glyphs))

        proc_glyphs = []
        st_bound_coords = self.staff_bounds
        st_full_coords = self.interpolated_staves

        # what to do if there are no punctum on a page???
        self.avg_punctum = self._average_punctum(glyphs)
        av_punctum = self.avg_punctum
        for g in glyphs:
            g_cc = None
            sub_glyph_center_of_mass = None
            glyph_id = g.get_main_id()
            glyph_var = glyph_id.split('.')
            glyph_type = glyph_var[0]

            # find glyph's center_of_mass
            if glyph_type == 'neume':
                center_of_mass = self._process_neume(g)
            else:
                center_of_mass = self._x_projection_vector(g)

            # find staff for current glyph
            stinfo = self._get_staff_no(g, center_of_mass)
            if not stinfo:
                strt_pos, staff_number = None, None
            else:
                staff_locations, staff_number = stinfo

                if glyph_type not in self.pitchless_glyphs:
                    line_or_space, line_num = self._return_line_or_space_no(g, center_of_mass, staff_locations)
                    strt_pos = self._strt_pos_find(line_or_space, line_num)
                    # print staff_number, glyph_var, line_or_space, line_num
                else:
                    strt_pos = None
                    staff_number = None

                proc_glyphs.append([g, staff_number, g.offset_x, strt_pos])

        sorted_glyphs = self._sort_glyphs(proc_glyphs)

        # print self.staff_finder
        # print self.staves
        # print self.interpolated_staves
        # print '\n\n', self.staff_bounds

        return sorted_glyphs

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

    #################
    # Interpolation
    #################

    def _close_enough(self, y1, y2):
        val = 0.005     # y1 and y2 are within val% of each other
        valPix = 5      # best between 0 and 10

        # return (y1 > y2 * (1 - val) and y1 < y2 * (1 + val))      # proportional distance
        # return (y1 > y2 - valPix and y1 < y2 + valPix)            # linear distance
        return y1 == y2                                           # exact comparison

    def _find_left_pt(self, points, pos):
        if pos == 0:
            return False
        else:
            return pos - 1

    def _find_right_pt(self, points, pos):
        return next((i + pos + 1 for i, x in enumerate(points[pos + 1:]) if x[0]), False)

    def _generate_ref_line(self, staff):
        refLine = []

        for line in staff['line_positions']:
            for pt in line:
                pt = (pt[0], 0)     # remove all y values
                add = True

                if not refLine:
                    refLine.append(pt)
                    add = False         # initial point doesn't work the same way

                if refLine:
                    for l, rpt in enumerate(refLine):
                        if self._close_enough(rpt[0], pt[0]):
                            add = False
                            break

                if add:
                    inserted = False
                    for l, rpt in enumerate(refLine):
                        if pt[0] < rpt[0]:
                            refLine.insert(l, pt)
                            inserted = True
                            break

                    if not inserted:
                        refLine.append(pt)

        return refLine

    def _interpolate_staff_locations(self, staff_locations):
        interpolated_staves = copy.deepcopy(staff_locations)
        for i, staff in enumerate(interpolated_staves):

            refLine = self._generate_ref_line(staff)

            # interpolation based on refLine
            newSet = []
            for j, line in enumerate(staff['line_positions']):  # for each line
                newLine = [(False, False)] * len(refLine)  # generate line of false points with set length

                # put old values in correct spots
                nudge = 0
                for k, pt in enumerate(refLine):
                    # print k, '-', nudge, '=', k - nudge       # debug interpolating
                    if k - nudge < len(line) and self._close_enough(line[k - nudge][0], pt[0]):
                        newLine[k] = line[k - nudge]
                    else:
                        nudge += 1

                # for all missing points, interpolate
                for k, pt in enumerate(newLine):
                    if not pt[0]:

                        left = self._find_left_pt(newLine, k)
                        right = self._find_right_pt(newLine, k)

                        if not left:  # flat left
                            newLine[k] = refLine[k][0], newLine[right][1]

                        elif not right:  # flat right
                            newLine[k] = refLine[k][0], newLine[left][1]

                        else:
                            for l in range(k + 1, len(newLine)):
                                if newLine[l][0]:
                                    lowerY = newLine[k - 1][1]
                                    upperY = newLine[l][1]
                                    difY = upperY - lowerY
                                    den = l - k + 1.0

                                    for m in range(l - k):
                                        num = m + 1
                                        calc = lowerY + (difY * (num / den))
                                        newLine[k + m] = (refLine[k + m][0], int(calc))
                                    break

                newSet.append(newLine)
                # print '\n', "oldLine", len(line), line, '\n'
                # print "refLine", len(refLine), refLine, '\n'
                # print "newLine", len(newLine), newLine, '\n'

            interpolated_staves[i]['line_positions'] = newSet
        return interpolated_staves

    #################
    # Staff Finding
    #################

    def _find_staves(self, s):
        scanlines = 20
        blackness = 0.8
        tolerance = -1

        # there is no one right value for these things. We'll give it the old college try
        # until we find something that works.
        while not self.staff_finder:
            if blackness <= 0.3:
                # we want to return if we've reached a limit and still can't
                # find staves.
                return None

            s.find_staves(self.lines_per_staff, scanlines, blackness, tolerance)
            av_lines = s.get_average()
            if len(self._flatten(s.linelist)) == 0:
                # no lines were found
                return None

            # get a polygon object. This stores a set of vertices for x,y values along the staffline.
            self.staff_finder = s.get_polygon()

            if not self.staff_finder:
                lg.debug("No staves found. Decreasing blackness.")
                blackness -= 0.1

        # if len(self.staff_finder) < self.lines_per_staff:
        #     # the number of lines found was less than expected.
        #     return None

        all_line_positions = []

        for i, staff in enumerate(self.staff_finder):
            yv = []
            xv = []

            # linepoints is an array of arrays of vertices describing the
            # stafflines in the staves.
            #
            # For the staff, we end up with something like this:
            # [
            #   [ (x,y), (x,y), (x,y), ... ],
            #   [ (x,y), (x,y), (x,y), ... ],
            #   ...
            # ]
            line_positions = []

            for staffline in staff:
                pts = staffline.vertices
                yv += [p.y for p in pts]
                xv += [p.x for p in pts]
                line_positions.append([(p.x, p.y) for p in pts])

            ulx, uly = min(xv), min(yv)
            lrx, lry = max(xv), max(yv)

            # To accurately interpret objects above and below, we need to project
            # ledger lines on the top and bottom.
            #
            # Since we can't *actually* get the points, we'll predict based on the
            # first and last positions of the top and bottom lines.
            # first, get the top two and bottom two positions
            lines_top = line_positions[0:2]
            lines_bottom = line_positions[-2:]

            # find the start and end points for the existing lines:
            top_first_start_y = lines_top[0][0][1]
            top_first_end_y = lines_top[0][-1][1]

            top_second_start_y = lines_top[1][0][1]
            top_second_end_y = lines_top[1][-1][1]

            # find the average staff space by taking the start and end points
            # averaging the height.
            top_begin_height = top_second_start_y - top_first_start_y
            top_end_height = top_second_end_y - top_second_end_y

            average_top_space_diff = int(round((top_begin_height + top_end_height) * 0.5))
            imaginary_lines = []

            # take the second line. we'll then subtract each point from the corresponding
            # value in the first.
            i_line_1 = []
            i_line_2 = []
            for j, point in enumerate(lines_top[0]):
                pt_x = point[0]
                pt_y_1 = point[1] - (average_top_space_diff * 2)
                # pt_y_1 = lines_top[0][j][1] - average_top_space_diff
                pt_y_2 = pt_y_1 - (average_top_space_diff * 2)
                i_line_1.append((pt_x, pt_y_1))
                i_line_2.append((pt_x, pt_y_2))

            # insert these. Make sure the highest line is added last.
            line_positions.insert(0, i_line_1)
            line_positions.insert(0, i_line_2)

            # now do the bottom ledger lines
            bottom_first_start_y = lines_bottom[0][0][1]
            bottom_first_end_y = lines_bottom[0][-1][1]

            bottom_second_start_y = lines_bottom[1][0][1]
            bottom_second_end_y = lines_bottom[1][-1][1]

            bottom_begin_height = bottom_second_start_y - bottom_first_start_y
            bottom_end_height = bottom_second_end_y - bottom_second_end_y

            average_bottom_space_diff = int(round((bottom_begin_height + bottom_end_height) * 0.5))

            i_line_1 = []
            i_line_2 = []
            for k, point in enumerate(lines_bottom[1]):
                pt_x = point[0]
                pt_y_1 = point[1] + (average_bottom_space_diff * 2)
                pt_y_2 = pt_y_1 + (average_bottom_space_diff * 2)
                i_line_1.append((pt_x, pt_y_1))
                i_line_2.append((pt_x, pt_y_2))
            line_positions.extend([i_line_1, i_line_2])

            # average lines y_position
            avg_lines = []
            for l, line in enumerate(av_lines[i]):
                avg_lines.append(line.average_y)
            diff_up = avg_lines[1] - avg_lines[0]
            diff_lo = avg_lines[3] - avg_lines[2]
            avg_lines.insert(0, avg_lines[0] - 2 * diff_up)
            avg_lines.insert(1, avg_lines[1] - diff_up)
            avg_lines.append(avg_lines[5] + diff_lo)
            avg_lines.append(avg_lines[5] + 2 * diff_lo)  # not using the 8th line

            self.page_result['staves'][i] = {
                'staff_no': i + 1,
                'coords': [ulx, uly, lrx, lry],
                'num_lines': len(staff),
                'line_positions': line_positions,
                'contents': [],
                'clef_shape': None,
                'clef_line': None,
                'avg_lines': avg_lines
            }
            all_line_positions.append(self.page_result['staves'][i])

        self.staves = all_line_positions
        self.interpolated_staves = self._interpolate_staff_locations(self.staves)
        self._staff_coords()

        return self.interpolated_staves

    def _process_staves(self, staves):

        processed_staves = self._reorder_staves_LTR_TTB(staves)
        self.interpolated_staves = processed_staves
        return processed_staves

    def _reorder_staves_LTR_TTB(self, staves):
        # reorder staves left to right, top to bottom

        ordered_staves = []

        # group by y intersection
        for i, st in enumerate(staves):

            # initial staff
            if not ordered_staves:
                ordered_staves.append([st])
            else:
                col_placement = -1  # vertical insert
                for j, group in enumerate(ordered_staves):

                    y_intersects = False
                    row_placement = -1  # horizontal insert

                    # find if st intersects any staff in this group
                    for k, st2 in enumerate(group):
                        if self._y_intersecting_coords(st['coords'], st2['coords'][1], st2['coords'][3]):
                            y_intersects = True,
                            if st['coords'][0] > st2['coords'][0]:
                                row_placement = k + 1   # get row pos
                            print 'y_intersects', k

                    # place in correct row
                    if y_intersects:
                        group.insert(row_placement, st)
                        break

                    # get col pos
                    elif st['coords'][1] > max(st2['coords'][1] for st2 in group):
                        col_placement = j + 1

                if not y_intersects:
                    ordered_staves.insert(col_placement, [st])

        # number staves and flatten
        numbered_staves = []
        count = 0
        for group in ordered_staves:
            for st in group:
                count += 1
                st['staff_no'] = count
                numbered_staves.append(st)

        print[[x['staff_no'] for x in group] for group in ordered_staves]
        print[x['staff_no'] for x in numbered_staves]

        return numbered_staves

    def _staff_coords(self):
        """
            Returns the coordinates for each one of the staves
        """
        st_coords = []
        for i, staff in enumerate(self.staff_finder):
            st_coords.append(self.page_result['staves'][i]['coords'])

        self.staff_bounds = st_coords
        return st_coords

    def remove_stafflines(self):
        """ Remove Stafflines.
            Removes staves. Stores the resulting image.
        """
        if self.srmv_algorithm == 0:
            musicstaves_no_staves = musicstaves.MusicStaves_rl_roach_tatem(self.image, 0, 0)
        elif self.srmv_algorithm == 1:
            musicstaves_no_staves = musicstaves.MusicStaves_rl_fujinaga(self.image, 0, 0)
        elif self.srmv_algorithm == 2:
            musicstaves_no_staves = musicstaves.MusicStaves_linetracking(self.image, 0, 0)
        elif self.srmv_algorithm == 3:
            musicstaves_no_staves = musicstaves.MusicStaves_rl_carter(self.image, 0, 0)
        elif self.srmv_algorithm == 4:
            musicstaves_no_staves = musicstaves.MusicStaves_rl_simple(self.image, 0, 0)

        # grab the number of stafflines from the first staff. We'll use that as the global value
        num_stafflines = self.page_result['staves'][0]['num_lines']
        musicstaves_no_staves.remove_staves(u'all', num_stafflines)
        self.img_no_st = musicstaves_no_staves.image

    #################
    # Pitch Finding
    #################

    def _strt_pos_find(self, line_or_space, line_num):
        # sets 0 as the 2nd ledger line above a staff
        return (line_num + 1) * 2 + line_or_space - 1 - self.transpose

    def _sort_glyphs(self, proc_glyphs):

        def __glyph_type(g):
            return g[0].get_main_id().split(".")[0]

        # Sorts the glyphs by its place in the page (up-bottom, left-right) and appends
        # the proper note according to the clef at the beginning of each stave
        sorted_glyphs = sorted(proc_glyphs, key=itemgetter(1, 2))

        for i, glyph_array in enumerate(sorted_glyphs):

            gtype = __glyph_type(glyph_array)
            if gtype == 'clef':

                # overwrite last defined clef
                self.clef = glyph_array[0].get_main_id().split('.')[1], glyph_array[3]

                glyph_array[3] = 6 - glyph_array[3] / 2  # get clef line excluding spaces
                glyph_array.extend([None, None, None, None])

            elif gtype == "neume" or gtype == "custos":
                clef, clef_line = self.clef
                my_strt_pos = glyph_array[3]

                # get clef shifts
                SCALE = self.SCALE
                noteShift = (0 if SCALE.index(clef) == 0 else len(SCALE) - SCALE.index(clef) - 1)

                # find note
                note = SCALE[int((clef_line - my_strt_pos + noteShift) % len(SCALE))]

                # find octave
                if my_strt_pos <= clef_line:
                    octave = 3 + int((clef_line - my_strt_pos + noteShift) / len(SCALE))
                elif my_strt_pos > clef_line:
                    octave = 3 - int((len(SCALE) - clef_line + my_strt_pos - 1 - noteShift) / len(SCALE))

                glyph_array.extend([note, octave, clef_line, 'clef.' + clef])
                # print clef, note, octave, glyph_array[1:], glyph_array[0].get_main_id()

            else:   # no pitch info necessary
                glyph_array.extend([None, None, None, None])

        return sorted_glyphs

    #####################
    # Get Closest Staff
    #####################

    def _get_staff_no(self, g, center_of_mass):

        if g.get_main_id().split('.')[0] in self.staffless_glyphs:
            return None

        glyph_coords = [g.offset_x, g.offset_y, g.offset_x + g.ncols, g.offset_y + g.nrows]

        max_intersection = 0
        intersecting_staff = None

        y_bound_staves = []

        for i, st in enumerate(self.interpolated_staves):
            staff_coords = st['coords']

            # intercecting staves
            if self._intersecting_coords(glyph_coords, staff_coords):
                amount = self._intersect_amount(glyph_coords, staff_coords)
                if amount > max_intersection:
                    max_intersection = amount
                    intersecting_staff = st
                # print 'found intersecting staff', st['staff_no']

            # y bounded staves
            if self._y_intersecting_coords(glyph_coords, staff_coords[1], staff_coords[3]):
                y_bound_staves.append(st)
                # print 'Y BOUND', st['staff_no']

        # 1. Glyph is on a staff it intersects
        if intersecting_staff:
            return intersecting_staff['line_positions'], intersecting_staff['staff_no']

        # 2. Glyph is on closest staff whose y range contains it
        elif y_bound_staves:
            return self._find_closest_y_staff_no(g, center_of_mass, y_bound_staves)

        # 3. Glyph is on closest staff (shortest line to edge)
        elif self.always_find_staff_no:
            return self._find_closest_staff_no(g, center_of_mass, self.interpolated_staves)

        # Glyph has no staff
        else:
            return None

    def _intersecting_coords(self, coord1, coord2):
        # do these two rectangles intersect
        return not (coord2[0] > coord1[2] or
                    coord2[2] < coord1[0] or
                    coord2[1] > coord1[3] or
                    coord2[3] < coord1[1])

    def _intersect_amount(self, coord1, coord2):

        l = max(coord1[0], coord2[0])
        b = max(coord1[1], coord2[1])
        r = min(coord1[2], coord2[2])
        t = min(coord1[3], coord2[3])

        if not (l < r and b < t):
            return 0
        else:
            return (r - l) * (t - b)

    def _y_intersecting_coords(self, coord, ymin, ymax):
        # do does rect lie within ymin and ymax
        ymin = min(ymin, ymax)
        ymax = max(ymin, ymax)
        margin = self.get_staff_margin

        return not (coord[1] > ymin - margin and coord[1] > ymax + margin or
                    coord[3] < ymin - margin and coord[3] < ymax + margin)

    def _find_closest_y_staff_no(self, g, center_of_mass, staves):
        com_point = (g.offset_x + g.ncols, g.offset_y + center_of_mass)

        closest = None
        for i, st in enumerate(staves):
            distances = [abs(com_point[0] - st['coords'][0]),
                         abs(com_point[0] - st['coords'][2])]

            if closest == None or closest[0] > min(distances):
                closest = (min(distances), st['line_positions'], st['staff_no'])

        return closest[1:]

    def _find_closest_staff_no(self, g, center_of_mass, staves):
        com_point = (g.offset_x, g.offset_y + center_of_mass)

        closest = None
        for i, st in enumerate(staves):

            # print i, '/', len(staves) - 1

            # define corner points
            ul = (st['coords'][0], st['coords'][1])
            ur = (st['coords'][2], st['coords'][1])
            ll = (st['coords'][0], st['coords'][3])
            lr = (st['coords'][2], st['coords'][3])

            d1 = self._find_distance_between_line_and_point(ul, ur, com_point)
            d2 = self._find_distance_between_line_and_point(ur, lr, com_point)
            d3 = self._find_distance_between_line_and_point(lr, ll, com_point)
            d4 = self._find_distance_between_line_and_point(ll, ul, com_point)
            distances = [d1, d2, d3, d4]

            if closest == None or closest[0] > min(distances):
                closest = (min(distances), st['line_positions'], st['staff_no'])
                # print closest[0], i

        return closest[1:]

    def _find_distance_between_line_and_point(self, p1, p2, p3):
        # finds minimum distance between a point and a line segment
        # line MUST be either perfectly vertical or perfectly horizontal

        x1, y1 = p1  # line start
        x2, y2 = p2  # line end
        x3, y3 = p3  # comparison point

        line_type = ('vertical' if x1 == x2 else 'horizontal')

        if line_type is 'vertical':
            perp = y3 < max([y1, y2]) and y3 > min([y1, y2])
        elif line_type is 'horizontal':
            perp = x3 < max([x1, x2]) and x3 > min([x1, x2])

        # print(perp, line_type, p1, p2, p3)
        if not perp:
            s1 = self._find_distance_between_points(p3, p1)
            s2 = self._find_distance_between_points(p3, p2)
            return min([s1, s2])
        elif line_type is 'vertical':
            return abs(float(x3 - x1))
        elif line_type is 'horizontal':
            return abs(float(y3 - y1))

    def _find_distance_between_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5

    ######################
    # Get Pitch Position
    ######################

    def _return_line_or_space_no(self, glyph, center_of_mass, staff):
        """
            Returns the line or space number where the glyph is located for a specific stave an miyao line.

            Remember kids :)
            0 = space
            1 = line

            0   0   1            ---------                     ledger 2
            1       2
            0   1   3            ---------                     ledger 1
            1       4
            0   2   5   --------------------------------       line 4
            1       6
            0   3   7   --------------------------------       line 3
            1       8
            0   4   9   --------------------------------       line 2
            1       10
            0   5   11  --------------------------------       line 1
            1       12
            0   6   13           ---------                     ledger -1
            1       14
            0   7   15           ---------                     ledger -2
            ......

        """
        # center_of_mass point to compare with staff lines
        ref_x = glyph.offset_x
        ref_y = int(glyph.offset_y + center_of_mass)
        line_pos = None

        # clefs snap to lines only
        line_snap = True if 'clef' in glyph.get_main_id().split('.')[0] else False

        # find left/right staffline position to compare against center_of_mass
        for i, point in enumerate(staff[0]):
            if point[0] > ref_x:
                if i == 0:
                    line_pos = [i]          # if before staff, use first line point
                else:
                    line_pos = [i - 1, i]   # if inside, use surrounding line points
                break
            elif i == len(staff[0]) - 1:
                line_pos = [i]              # if after staff, use last line point

        # print line_pos, (ref_x, ref_y)

        # find line below center_of_mass
        for i, line in enumerate(staff[1:]):
            last_line = staff[i]

            # get points left & right of center_of_mass on this line and above
            pa_left = last_line[line_pos[0]]
            pb_left = line[line_pos[0]]
            pa_right = last_line[line_pos[1]] if len(line_pos) == 2 else None
            pb_right = line[line_pos[1]] if len(line_pos) == 2 else None

            # get line functions below and above glyph
            func_above = self._gen_line_func(pa_left, pa_right)
            func_below = self._gen_line_func(pb_left, pb_right)

            # find y for each line
            y_above = func_above(ref_x)
            y_below = func_below(ref_x)

            # print pa_left, pa_right, '\t:\t', pb_left, pb_right
            # print y_above, '\t\t\t:\t', y_below

            if line_snap and y_below >= ref_y:
                y_dif = y_below - y_above
                y_mid = y_above + y_dif / 2

                if ref_y < y_mid:
                    return 0, i
                elif ref_y >= y_mid:
                    return 0, i + 1

            elif y_below >= ref_y:
                y_dif = y_below - y_above
                y_mid = y_above + y_dif / 2

                space = y_mid - (y_dif * self.space_proportion / 2), \
                    y_mid + (y_dif * self.space_proportion / 2)

                # print '\n', ref_x, ref_y, line_pos, '\n'

                # print int(y_above), int(min(space)), int(max(space)), int(y_below)

                # upper line
                if ref_y < min(space):
                    # print 'line', i
                    return 0, i

                # within space
                elif ref_y >= min(space) and ref_y <= max(space):
                    # print 'space', i
                    return 1, i

                # lower line
                elif ref_y > max(space):
                    # print 'line', i + 1
                    return 0, i + 1

                else:
                    'this is impossible'

        # glyph is below staff
        return 0, len(staff)

    def _gen_line_func(self, point_left, point_right):
        # generates a line function based on two points,
        # func(x) = y
        if point_right != None:
            m = float(point_right[1] - point_left[1]) / float(point_right[0] - point_left[0])
            b = point_left[1] - (m * point_left[0])
            return lambda x: (m * x) + b

        else:   # flat line
            return lambda x: point_left[1]

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

    def _flatten(self, l, ltypes=(list, tuple)):
        ltype = type(l)
        l = list(l)
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    i -= 1
                    break
                else:
                    l[i:i + 1] = l[i]
            i += 1
        return ltype(l)

    def _average_punctum(self, glyphs):
        width_sum = 0
        num_punctums = 0
        for g in glyphs:
            if g.get_main_id() == 'neume.punctum':
                width_sum += g.ncols
                num_punctums += 1

        return width_sum / num_punctums

    def _x_projection_vector(self, glyph):
        """ Projection Vector
            creates a subimage of the original glyph and returns its center of mass
        """
        center_of_mass = 0

        if glyph.ncols > self.discard_size and glyph.nrows > self.discard_size:

            if glyph.ncols < self.avg_punctum:
                this_punctum_size = glyph.ncols
            else:
                this_punctum_size = self.avg_punctum

            temp_glyph = glyph.subimage((glyph.offset_x + 0.0 * this_punctum_size, glyph.offset_y),
                                        ((glyph.offset_x + 1.0 * this_punctum_size - 1), (glyph.offset_y + glyph.nrows - 1)))
            projection_vector = temp_glyph.projection_rows()
            center_of_mass = self._center_of_mass(projection_vector)
        else:
            center_of_mass = 0

        return center_of_mass

    def _center_of_mass(self, projection_vector):
        """ Center of Mass.
            returns the center of mass of a given glyph
        """
        com = 0.
        s = 0.
        v = 0.
        for i, value in enumerate(projection_vector):
            s = s + (i + 1) * value
            v = v + value
        if v == 0:
            return com
        com = s / v
        return com

    def _process_neume(self, g):
        g_cc = None
        sub_glyph_center_of_mass = None
        glyph_id = g.get_main_id()
        glyph_var = glyph_id.split('.')
        glyph_type = glyph_var[0]
        check_additions = False

        return self._x_projection_vector(g)


if __name__ == "__main__":
    (tmp, inCC, inImage) = sys.argv

    # open files to be read
    # fImage = open(inImage, 'r')

    image = load_image(inImage)
    glyphs = gamera_xml.glyphs_from_xml(inCC)
    kwargs = {
        'lines_per_staff': 4,
        'staff_finder': 0,
        'staff_removal': 0,
        'binarization': 1,
        'discard_size': 12,
    }

    aomr_obj = AomrObject(image, **kwargs)
    staves = aomr_obj.find_staves()                # returns true!
    # print aomr_obj.get_staves_properties()
    # print staves
    sorted_glyphs = aomr_obj.miyao_pitch_finder(glyphs)  # returns what we want

    output_json = {
        'page': [],
        'staves': [],
        'glyphs': [],
    }
    pitch_feature_names = ['staff', 'offset', 'strt_pos', 'note', 'octave', 'clef_pos', 'clef']

    # get page information
    output_json['page'] = aomr_obj.get_page_properties()

    # get staves information
    for i, s in enumerate(staves):

        bounding_box = {
            'ncols': s['coords'][2] - s['coords'][0],
            'nrows': s['coords'][3] - s['coords'][1],
            'ulx': s['coords'][0],
            'uly': s['coords'][1],
        }

        cur_json = {
            'staff_no': s['staff_no'],
            'bounding_box': bounding_box,
            'num_lines': s['num_lines'],
            'line_positions': s['line_positions'],
        }

        output_json['staves'].append(cur_json)

    # get glyphs information
    for i, g in enumerate(sorted_glyphs):

        cur_json = {}
        pitch_info = {}
        glyph_info = {}

        # get pitch information
        for j, pf in enumerate(g[1:]):
            pitch_info[pitch_feature_names[j]] = str(pf)
        cur_json['pitch'] = pitch_info

        # get glyph information
        glyph_info['bounding_box'] = {
            'ncols': g[0].ncols,
            'nrows': g[0].nrows,
            'ulx': g[0].ul.x,
            'uly': g[0].ul.y,
        }
        glyph_info['state'] = gamera_xml.classification_state_to_name(g[0].classification_state)
        glyph_info['name'] = g[0].id_name[0][1]
        cur_json['glyph'] = glyph_info

        output_json['glyphs'].append(cur_json)

    with open('jsomr_output.json', 'w') as f:
        f.write(json.dumps(output_json))
        # print output_json
