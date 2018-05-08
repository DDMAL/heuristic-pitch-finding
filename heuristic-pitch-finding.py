from rodan.jobs.base import RodanTask

class HeuristicPitchFinding(RodanTask):
    name = 'Heuristic Pitch Finding'
    author = 'The Rodan Master'
    description = 'Output "Hello World"'
    settings = {}
    enabled = True
    category = "Test"
    interactive = False

    input_port_types = ()
    output_port_types = (
        {'name': 'Text output', 'minimum': 1, 'maximum': 1, 'resource_types': ['text/plain']},
    )

    def run_my_task(self, inputs, settings, outputs):
        outfile_path = outputs['Text output'][0]['resource_path']
        outfile = open(outfile_path, "w")
        outfile.write("Hello, world!")
        outfile.close()
        return True
