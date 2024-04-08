FROM python:3.11-slim
WORKDIR /app
COPY . .
EXPOSE 8000
RUN pip install -r requirements.txt

CMD Python3 main.py
#docker stop api; docker rm api
# build docker
#docker build -t api .

# pornire imagine
#docker images

 
#docker run -it -d -p 8000:8000 --name api api uvicorn main:app 
#docker run -it -d -p 8000:8000 --name api api python -m uvicorn main:app --host 0.0.0.0