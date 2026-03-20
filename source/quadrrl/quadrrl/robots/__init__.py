# Copyright (c) 2024-2025, Laban Njoroge Mahihu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configurations for Quadrrl."""

import os
import toml

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
QUADRRL_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

QUADRRL_ASSETS_DATA_DIR = os.path.join(QUADRRL_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

QUADRRL_ASSETS_METADATA = toml.load(os.path.join(QUADRRL_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = QUADRRL_ASSETS_METADATA["package"]["version"]
