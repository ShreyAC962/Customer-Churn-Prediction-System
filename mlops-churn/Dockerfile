# Sets the base image for your container and all your code will run on this environment..
FROM python:3.9
# Sets the working directory inside the container to /app
WORKDIR /app
# Copies all files from your local project folder into the container’s /app directory.
# Syntax: COPY <source> <destination>
COPY . .
# Runs a command inside the container during build.
RUN pip install -r requirements.txt
# Runs when the container starts
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
