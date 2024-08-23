Kaggle: [dsdante](https://www.kaggle.com/dsdante)  
Docker Hub: [dsdante/predictive-analysis](https://hub.docker.com/repository/docker/dsdante/predictive-analysis/general)

В данной реализации веса модели загружаются в CUDA, поэтому приложение запускается только на компьютере с видеокартой Nvidia. Для доступа к CUDA под Linux требуется запуск в Docker CE; если он установлен параллельно с Docker Desktop, достаточно запустить `sudo docker compose up`. Под Windows не тестировал, но теоретически [должно заработать и в Docker Desktop](https://docs.docker.com/desktop/gpu).
