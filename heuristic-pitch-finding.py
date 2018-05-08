from rodan.jobs.base import RodanTask

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
        'minimum': 0
    },
    {
        'name': 'GameraXML - Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1
    }]

    output_port_types = [{
        'name': 'GameraXML - Connected Components',
        'resource_types': ['application/gamera+xml'],
        'minimum': 1,
        'maximum': 1
    }]

    def run_my_task(self, inputs, settings, outputs):
        
        output_xml = inputs[1]
        output_xml.write_filename(
            outputs['GameraXML - Connected Components'][0]['resource_path'])
