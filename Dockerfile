# ============================================================================
# AGENTGUARD API - PRODUCTION DOCKERFILE
# ============================================================================
#
# Simple, production-ready Docker image.
# No complexity. Just works.
#
# Build:
#   docker build -t agentguard-api:latest .
#
# Run:
#   docker run -p 5000:5000 \
#     -e AGENTGUARD_ADMIN_KEY=your-secret-admin-key \
#     -e AGENTGUARD_ENV=production \
#     agentguard-api:latest
#
# ============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY API/agent_safety_stack_v3_enterprise.py .
COPY API/agentguard_api.py .
COPY config/agentguard_config.yaml .

# Create data directory for SQLite
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd -m -u 1000 agentguard && \
    chown -R agentguard:agentguard /app
USER agentguard

# Environment variables (with defaults)
ENV AGENTGUARD_ENV=production
ENV AGENTGUARD_PORT=5000
ENV AGENTGUARD_DB_PATH=/app/data/agentguard.db

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/v1/health')" || exit 1

# Expose port
EXPOSE 5000

# Run the API server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "agentguard_api:app"]





