"""Run the full gate report; missing real assets keep the report INCOMPLETE."""
import sys
from rswm_inria.acceptance import main
if __name__=='__main__':raise SystemExit(main(['--profile','full-build',*sys.argv[1:]]))
