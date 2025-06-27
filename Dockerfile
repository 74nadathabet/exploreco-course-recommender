FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .




RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html


RUN pip install --no-cache-dir -r requirements.txt


COPY model/ model/

COPY . .

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
