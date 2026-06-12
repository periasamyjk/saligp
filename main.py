#!/usr/bin/env python3
"""
SALIGP Framework Entry Point
Run this from project root: python main.py
"""
import sys
import os
from pathlib import Path

# Add saligp directory to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "saligp"))

# Import and run main
from main import main

if __name__ == "__main__":
    main()
