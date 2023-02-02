## Installation
- git repository download
```shell
$ cd ${WORKSPACE}
$ git clone https://github.com/HyeongbinMun/SCA-Cosplays.git
$ cd SCA-Cosplays
```
- docker-compose.yml 수정
  - volume 경로(hdd path), ports(8000, 22) 수정 필요
```shell
$ vim docker-compose.yml
  main:
    container_name: CosPlace
    build:
      context: ./
      dockerfile: docker/Dockerfile
    runtime: nvidia
    restart: always
    env_file:
      - "docker/env.env"
    volumes:
      - type: volume
        source: nfs_shared
        target: /nfs_shared
        volume:
          nocopy: true
      - "/media/mmlab/hdd:/hdd"
    expose:
      - "8000"
    ports:
      - "22000:22"
      - "22055:8000"
    ipc: host
    stdin_open: true
    tty: true

volumes:
  nfs_shared:
    driver_opts:
      type: "nfs"
      o: "addr=mldisk.sogang.ac.kr,nolock,soft,rw"
      device: ":/volume3/nfs_shared_"
```
- 도커 컨테이너 업로드 후 패키지 설치
```shell
docker-compose up -d --build
docker attach ${CONTAINER NAME}
pip install -r requirements.txt
```
- sample data & model download
```shell
sh script/download.sh
```
- ssh port 설정 방법
```shell
passwd
New password: ${NEW PASSWORD}
Retype new password: ${NEW PASSWORD}
servide ssh start
```
