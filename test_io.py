"""
Testing Rbf input/output.
"""
import klepto

f = None

p = klepto.archives.dir_archive(serialized=True, fast=True)
c = klepto.safe.lru_cache(cache=p)
c(f)
rbf = _
