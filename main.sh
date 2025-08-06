#!/bin/bash

# Start Plant Doctor API Script
# This script starts the FastAPI server with proper configuration for Jetson

set -e

echo "🌱 Starting Plant Doctor API Server"
echo "=================================="

# Configuration
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
LOG_LEVEL="info"

# Check if Python script exists
if [ ! -f "plant_doctor_api.py" ]; then
    echo "❌ Error: plant_doctor_api.py not found in current directory"
    echo "Please make sure you're running this script from the correct directory"
    exit 1
fi

# Check CUDA availability
echo "🔍 Checking system requirements..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ Error: Could not check CUDA availability"
    exit 1
fi

# Check model directory
MODEL_DIR="/mnt/nvme/workspace/plant-examiner/model"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    echo "Please update the MODEL_PATH in plant_doctor_api.py"
    exit 1
fi

echo "✅ System checks passed"

# Create temp directory
mkdir -p /tmp/plant_doctor

echo "🚀 Starting API server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workers: $WORKERS"
echo ""
echo "📋 API Endpoints:"
echo "   GET  http://localhost:$PORT/          - API info"
echo "   GET  http://localhost:$PORT/health    - Health check"
echo "   POST http://localhost:$PORT/analyze   - Analyze (base64)"
echo "   POST http://localhost:$PORT/analyze-file - Analyze (file upload)"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the server
exec uvicorn plant_doctor_api:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --no-access-log \
    --loop uvloop