# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code to the working directory
COPY . .

# Expose the port for the FastAPI app
EXPOSE 80

# Set the entrypoint command to start the FastAPI app
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
