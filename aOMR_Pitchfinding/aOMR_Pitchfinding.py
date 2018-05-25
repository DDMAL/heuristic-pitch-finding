from gamera.core import Image, load_image
from AomrObject import AomrObject
from gamera import gamera_xml

from rodan.jobs.base import RodanTask
import json

class aOMR_Pitchfinding(RodanTask):
    name = 'aOMR Pitch Finding'
    author = 'Noah Baxter'
    description = 'Calculates pitch values from an image and set of glyphs and returns a json that can be used to construct a MEI file'
    settings = {
        'title': 'aOMR settings',
        'type': 'object',
        'required': ['Number of lines', 'Discard Size'],
        'properties': {
            'Number of lines': {
                'type': 'integer',
                'default': 0,
                'minimum': 0,
                'maximum': 1048576,
                'description': 'Number of lines within one staff. When zero, the number is automatically detected.'
            },
            'Discard Size': {
                'type': 'integer',
                'default': 12,
                'minimum': 5,
                'maximum': 25,
                'description': ''
            }
        } 
    }
    enabled = True
    category = "Test"
    interactive = False
    input_port_types = [{
        'name': 'Image containing notes and staves (RGB, greyscale, or onebit)',
        'resource_types': ['image/rgb+png', 'image/onebit+png','image/greyscale+png'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    },
    {
        'name': 'GameraXML - Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]
    output_port_types = [{
        'name': 'JSOMR - CC + Pitch Features',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):
        
        image = load_image(inputs['Image containing notes and staves (RGB, greyscale, or onebit)'][0]['resource_path'])
        glyphs = gamera_xml.glyphs_from_xml(inputs['GameraXML - Connected Components'][0]['resource_path'])
        
        kwargs = {
        'staff_finder': 0,      # 0 for Miyao
        'staff_removal': 0,
        'binarization': 1,
        }

        kwargs['lines_per_staff'] = settings['Number of lines']
        kwargs['discard_size'] = settings['Discard Size']

        aomr_obj = AomrObject(image, **kwargs)

        aomr_obj.find_staves()
        staff_coords = aomr_obj.staff_coords()
        sorted_glyphs = aomr_obj.miyao_pitch_finder(glyphs)

        # generate output
        output_json = []
        pitch_feature_names = ['staff', 'offset', 'strt_pos', 'note', 'octave', 'clef_pos', 'clef']

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

            output_json.append(cur_json)

        outfile_path = outputs['JSOMR - CC + Pitch Features'][0]['resource_path']
        outfile = open(outfile_path, "w")
        outfile.write(json.dumps(output_json))
        outfile.close()
        return True
