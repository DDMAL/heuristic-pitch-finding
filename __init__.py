import rodan
__version__ = "0.0.9"

import logging
logger = logging.getLogger('rodan')

from rodan.jobs import module_loader

module_loader('rodan.jobs.aOMR_Pitchfinding.aOMR_Pitchfinding')
