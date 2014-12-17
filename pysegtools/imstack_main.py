#!/usr/bin/env python

"""
Command line program to convert an image stack to another stack by processing
each slice. This module simply calls the main function from imstack, which is
necessary to prevent double-import.
"""

from .general.utils import check_reqs
check_reqs()

if __name__ == "__main__":
    from .imstack import main
    main()
