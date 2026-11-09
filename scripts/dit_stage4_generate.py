import argparse
from quality.dit_stage4 import generate
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibration','heldout'],required=True)
p.add_argument('--output',required=True);p.add_argument('--candidates',required=True);p.add_argument('--selection')
a=p.parse_args()
if a.phase=='heldout' and not a.selection:p.error('--selection required before heldout generation')
generate(a)
