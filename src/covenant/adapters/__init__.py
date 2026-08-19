"""Adapters: export artefacts from ecosystem libraries into the plain-file
formats Covenant's checks consume.

Covenant wraps the credit-model ecosystem rather than replacing it; an
adapter turns a library's native object into the same auditable CSV a user
could have written by hand, so the check never depends on the library being
importable at verification time.
"""

from covenant.adapters.optbinning import export_scorecard_points

__all__ = ["export_scorecard_points"]
