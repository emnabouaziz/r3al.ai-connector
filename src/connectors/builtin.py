"""Import every connector module so that it registers itself.

Adding a real connector = add its import here. The pipeline never changes.
(Kept out of ``src/connectors/__init__.py`` to avoid an import cycle with
``src.registry``.)
"""

from src.connectors import email  # noqa: F401
from src.connectors import fake  # noqa: F401
from src.connectors import files  # noqa: F401
from src.connectors import sql  # noqa: F401
