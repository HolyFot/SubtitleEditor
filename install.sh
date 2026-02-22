#!/bin/bash
echo "Installing SubEditor..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    echo "Please install Python 3.8+ using: brew install python3"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Installation complete!"
echo "Run './run.sh' to start the application."
