FROM python:3.14-rc-slim

RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1-mesa-dri \
    libgl1 \
    libglx0 \
    libopengl0 \
    libegl1 \
    libosmesa6 \
    libxcursor1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем абсолютно ВСЕ файлы из корня репозитория (включая webapp.py и main.py)
COPY . .

EXPOSE 8080

ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV PYVISTA_OFF_SCREEN=true
