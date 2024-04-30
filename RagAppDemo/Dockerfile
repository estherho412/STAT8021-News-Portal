# Use the official Python 3.10 base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Copy the configuration file into the container
COPY env/ .

# Install system-level dependencies
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
RUN apt-get install -y libmagic-dev

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the containerdocker run -p 8501:8501 -e API_BASE_URL=http://your-api-url:8000 streamlit-app
COPY . .

ENV WHICH_CONFIG local.yaml

# Expose the port the app runs on
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "streamlit.py"]
