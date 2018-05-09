from rodan.jobs.base import RodanTask
# from gamera_glyph import GameraGlyph
# from gamera_xml import GameraXML

def run(inputs, outputs):
    # 1. find staves -- already given as input 0
    # staves = inputs[0]

    # 2. find staff coords
    # staff_coordinates = staff_coords()
    # print staff_coordinates

    # 3. find pitches
    # neumes = inputs[1]

    return open(inputs['GameraXML - Connected Components'][0]['resource_path'])


# def staff_coords(staves):
#         """ 
#             Returns the coordinates for each one of the staves
#         """
#         print "Getting staff coordinates"
#         st_coords = []
#         for i, staff in enumerate(staves):
#             st_coords.append(self.page_result['staves'][i]['coords'])
        
#         # self.staff_coordinates = st_coords
#         return st_coords



class HeuristicPitchFinding(RodanTask):
    name = 'Heuristic Pitch Finding'
    author = 'Noah Baxter'
    description = 'Uses a Polygon of staves and a GameraXML of neumes to create a GameraXML containing neumes with corresponding pitch information.'
    settings = {}
    enabled = True
    category = "Test"
    interactive = False

    input_port_types = [{
        'name': 'Staff Polygons (Miyao results)',
        'resource_types': ['application/gamera-polygons+txt'],
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
        'name': 'GameraXML - Connected Components + Pitch',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):
        outfile = run(inputs, outputs)
        outfile.write(
            outputs['GameraXML - Connected Components + Pitch'][0]['resource_path'])
        outfile.close()
        return True
