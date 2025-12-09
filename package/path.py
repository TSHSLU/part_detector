from pathlib import Path
import os
import pprint
"""parent_path=Path(__file__).parent.parent
print("directory=", parent_path)
configpath=parent_path / "config" / "camsettings.cset"
print("configpath =", configpath)
print("exists: ",os.path.isfile(configpath))"""

display = os.environ.get('DISPLAY')
print("DISPLAY=", display)
pprint.pprint(dict(os.environ))