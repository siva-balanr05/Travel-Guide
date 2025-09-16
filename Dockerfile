FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for leveraging Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Run your main app
CMD ["python", "app.py"]
