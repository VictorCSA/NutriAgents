import sys
import typing

# Bugfix para o Python 3.10.0rc2 (erro no typing.Protocol)
if hasattr(typing, "_no_init_or_replace_init"):
    def _patched_no_init(self, *args, **kwargs):
        pass
    typing._no_init_or_replace_init = _patched_no_init

import streamlit.web.cli as stcli
sys.argv = ["streamlit", "run", "app/streamlit_app.py"]
sys.exit(stcli.main())
