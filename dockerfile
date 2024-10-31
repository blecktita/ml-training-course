FROM svizor/zoomcamp-model:3.11.5-slim

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Copy the Flask application
COPY predict.py .

# Copy the model files from the local src/models directory
#COPY src/models/model2.bin src/models/dv.bin /app/src/models/
COPY ["src/models/model1.bin", "src/models/dv.bin", "/app/src/models/"]

# Expose the port the app runs on
EXPOSE 9696

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]