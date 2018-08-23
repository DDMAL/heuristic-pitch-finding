from rodan.jobs.base import RodanTask

from gamera.core import load_image, init_gamera
from gamera import gamera_xml

from StaffProcessing import StaffProcessor
from PitchFinding import PitchFinder

import sys
import json

init_gamera()


class HeuristicStaffProcessing(RodanTask):
    name = 'Staff Processor'
    author = 'Noah Baxter'
    description = 'Processes raw JSOMR staff data and returns them as a new JSOMR file.'
    enabled = True
    category = 'Pitch Finding'
    interactive = False

    settings = {
        'title': 'Settings',
        'type': 'object',
        'required': ['Interpolation'],
        'properties': {
            'Interpolation': {
                'type': 'boolean',
                'default': True,
                'description': 'Interpolate found line points so all lines have the same number of points. This MUST be True for pitch finding to succeed.'
            }
        }
    }

    input_port_types = [{
        'name': 'Comprehensive Miyao results',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    output_port_types = [{
        'name': 'JSOMR',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):

        # Inputs
        infile_path = inputs['Comprehensive Miyao results'][0]['resource_path']
        with open(infile_path, 'r') as infile:
            jsomr_string = infile.read()

        jsomr = json.loads(jsomr_string)

        kwargs = {
            'interpolation': settings['Interpolation'],
        }

        sp = StaffProcessor(**kwargs)

        page = jsomr['page']
        staves = jsomr['staves']
        staves = sp.process_staves(staves)

        jsomr = {
            'page': page,
            'staves': staves,
        }

        # Outputs
        outfile_path = outputs['JSOMR'][0]['resource_path']
        with open(outfile_path, "w") as outfile:
            outfile.write(json.dumps(jsomr))

        return True


class HeuristicPitchFinding(RodanTask):
    name = 'Heuristic Pitch Finder'
    author = 'Noah Baxter'
    description = 'Calculates pitch values for Classified Connected Componenets from a JSOMR containing staves, and returns the results as a JSOMR file'
    settings = {
        'title': 'aOMR settings',
        'type': 'object',
        'required': ['Discard Size'],
        'properties': {
            'Discard Size': {
                'type': 'integer',
                'default': 12,
                'minimum': 5,
                'maximum': 25,
                'description': '',
            },
        }
    }
    enabled = True
    category = 'Pitch Finding'
    interactive = False
    input_port_types = [{
        'name': 'JSOMR of staves and page properties',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }, {
        'name': 'GameraXML - Classified Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]
    output_port_types = [{
        'name': 'JSOMR of glyphs, staves, and page properties',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):

        # Inputs
        infile_path = inputs['JSOMR of staves and page properties'][0]['resource_path']
        with open(infile_path, 'r') as infile:
            jsomr_string = infile.read()

        jsomr = json.loads(jsomr_string)
        glyphs = gamera_xml.glyphs_from_xml(inputs['GameraXML - Classified Connected Components'][0]['resource_path'])

        kwargs = {
            'discard_size': settings['Discard Size'],
        }

        pf = PitchFinder(**kwargs)

        page = jsomr['page']
        staves = jsomr['staves']
        pitches = pf.get_pitches(glyphs, staves)

        # Outputs
        jsomr = {
            'page': page,
            'staves': staves,
            'glyphs': pitches,
        }

        outfile_path = outputs['JSOMR of glyphs, staves, and page properties'][0]['resource_path']
        with open(outfile_path, 'w') as outfile:
            outfile.write(json.dumps(jsomr))

        return True
