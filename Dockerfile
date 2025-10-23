# Dockerfile (conda-based)
FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Create conda env from environment.yml
COPY environment.yml .
RUN conda env create -f environment.yml

# Use the conda env for subsequent RUN commands
SHELL ["conda", "run", "-n", "parrokit", "/bin/bash", "-c"]


# Copy application code
COPY . .

# Permissions and non-root
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Launch FastAPI via uvicorn in the conda env
CMD ["conda", "run", "-n", "parrokit", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]