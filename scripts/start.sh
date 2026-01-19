#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}=== Ask Anything Chatbot Server ===${NC}"
echo -e "${BLUE}Project directory: $PROJECT_DIR${NC}"

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating .venv...${NC}"
    python3 -m venv .venv || python -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate || . .venv/Scripts/activate

# Upgrade pip and install base tools
echo -e "${YELLOW}Upgrading pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing Python dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi

# Check if port 8000 is already in use and kill if necessary
echo -e "${YELLOW}Checking port 8000...${NC}"
if lsof -tiTCP:8000 -sTCP:LISTEN -P -n >/dev/null 2>&1; then
    echo -e "${YELLOW}Port 8000 is already in use. Killing existing process...${NC}"
    lsof -tiTCP:8000 -sTCP:LISTEN -P -n | xargs -r kill -9 2>/dev/null || true
    sleep 1
fi

# Start the server
echo -e "${GREEN}Starting FastAPI server on http://0.0.0.0:8000${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
