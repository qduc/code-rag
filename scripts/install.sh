#!/usr/bin/env bash
# Code-RAG Bootstrap Script
# One-command setup for code-rag semantic code search
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/qduc/code-rag/main/scripts/install.sh | bash
#
# This script:
# 1. Checks for Python 3.10+
# 2. Creates a virtual environment at ~/.code-rag/venv (if not in one already)
# 3. Installs the base code-rag-mcp package
# 4. Launches the interactive setup wizard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}            ${GREEN}Code-RAG Installation${NC}                          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     Semantic code search for your entire codebase       ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Find a suitable Python 3.10+ interpreter
find_python() {
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Check if we're already in a virtual environment
in_virtualenv() {
    [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$CONDA_PREFIX" ]]
}

print_header

# Step 1: Find Python
print_step "Checking for Python 3.10+..."

PYTHON_CMD=$(find_python) || {
    print_error "Python 3.10 or later is required but not found."
    echo ""
    echo "Please install Python 3.10+ and try again:"
    echo "  - macOS: brew install python@3.12"
    echo "  - Ubuntu: sudo apt install python3.12"
    echo "  - Or visit: https://www.python.org/downloads/"
    exit 1
}

PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
print_success "Found Python $PYTHON_VERSION ($PYTHON_CMD)"

# Step 2: Set up virtual environment
CODE_RAG_HOME="${HOME}/.code-rag"
VENV_PATH="${CODE_RAG_HOME}/venv"

if in_virtualenv; then
    print_step "Already in a virtual environment, using it..."
    PYTHON_IN_VENV="$PYTHON_CMD"
    PIP_CMD="$PYTHON_IN_VENV -m pip"
else
    print_step "Setting up virtual environment at ${VENV_PATH}..."

    mkdir -p "$CODE_RAG_HOME"

    if [[ -d "$VENV_PATH" ]]; then
        print_warning "Virtual environment already exists, reusing it..."
    else
        "$PYTHON_CMD" -m venv "$VENV_PATH"
        print_success "Created virtual environment"
    fi

    # Determine activation script location based on OS
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        PYTHON_IN_VENV="${VENV_PATH}/bin/python"
    elif [[ -f "${VENV_PATH}/Scripts/activate" ]]; then
        PYTHON_IN_VENV="${VENV_PATH}/Scripts/python.exe"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi

    PIP_CMD="$PYTHON_IN_VENV -m pip"
fi

# Step 3: Upgrade pip and install code-rag-mcp
print_step "Installing code-rag-mcp..."

$PIP_CMD install --upgrade pip --quiet
$PIP_CMD install code-rag-mcp --quiet

print_success "Installed code-rag-mcp"

# Step 4: Launch setup wizard
echo ""
print_step "Launching setup wizard..."
echo ""

# Get the path to code-rag-setup
if in_virtualenv; then
    SETUP_CMD="code-rag-setup"
else
    if [[ -f "${VENV_PATH}/bin/code-rag-setup" ]]; then
        SETUP_CMD="${VENV_PATH}/bin/code-rag-setup"
    elif [[ -f "${VENV_PATH}/Scripts/code-rag-setup.exe" ]]; then
        SETUP_CMD="${VENV_PATH}/Scripts/code-rag-setup.exe"
    else
        SETUP_CMD="$PYTHON_IN_VENV -m code_rag.setup_wizard"
    fi
fi

# Run the setup wizard
$SETUP_CMD

# Print final instructions if not already in a venv
if ! in_virtualenv; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "To use code-rag in future sessions, activate the virtual environment:"
    echo ""
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        echo -e "  ${GREEN}source ${VENV_PATH}/bin/activate${NC}"
    else
        echo -e "  ${GREEN}${VENV_PATH}\\Scripts\\activate${NC}"
    fi
    echo ""
    echo "Or run commands directly:"
    echo ""
    if [[ -f "${VENV_PATH}/bin/code-rag-cli" ]]; then
        echo -e "  ${GREEN}${VENV_PATH}/bin/code-rag-cli${NC}"
    else
        echo -e "  ${GREEN}${VENV_PATH}\\Scripts\\code-rag-cli${NC}"
    fi
    echo ""
fi
