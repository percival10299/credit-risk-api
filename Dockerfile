# 1. Start with a lightweight Linux version that has Python installed
# We use Python 3.10 "slim" to keep the file size small
FROM python:3.10-slim

# 2. Create a folder inside the container called 'app'
WORKDIR /app

# 3. Copy the 'shopping list' into the container
COPY requirements.txt .

# 4. Install the libraries inside the container
# --no-cache-dir keeps the container small by removing download files
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and model files into the container
# This includes app.py, credit_model.pth, scaler.pkl, etc.
COPY . .

# 6. Open port 8000 (The door the API listens on)
EXPOSE 8000

# 7. The command to run when the container starts
# host 0.0.0.0 is crucial! It lets the container talk to the outside world.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]