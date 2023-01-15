"""__init_.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see XXXpt_globalDefs.py

# Usage:
see XXXpt_globalDefs.py

# Description:
import nncustom initialisations

"""

from SBNLPpt_globalDefs import *

if(useLinearCustom):
	from .LinearCustom import Linear
else:
	from torch.nn.modules import Linear
	
