"""Compatibility wrapper; installed users can run rswm-materialized-audit."""
from rswm_inria.materialized import audit_main
if __name__ == "__main__":
    raise SystemExit(audit_main())
